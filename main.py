"""
Team Kanyarasi — Sentinel-2 Farm Analysis API

Endpoints:

GET  /health
POST /analyze
GET  /ndvi-history?lat=<lat>&lon=<lon>

The /analyze endpoint returns the latest suitable Sentinel-2
scene and runs the Random Forest land-cover model.

The /ndvi-history endpoint retrieves real historical Sentinel-2
observations for the selected coordinate and calculates NDVI
from the Red (B04) and NIR (B08) bands.
"""

import base64
import io
from datetime import date, timedelta

import joblib
import numpy as np
import planetary_computer
import pystac_client
import rioxarray

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model_india_small.pkl"

CATALOG_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Latest analysis search range
DATE_RANGE = "2025-08-01/2026-08-20"

# Historical NDVI period
HISTORY_DAYS = 365

# Number of historical monthly observations returned
HISTORY_POINTS = 12

# Maximum acceptable scene-level cloud cover
MAX_CLOUD = 30

# Approx. 3 km x 3 km analysis window
BUFFER_DEG = 0.03


# ============================================================
# WORLDCOVER CLASSES
# ============================================================

WORLDCOVER_CLASSES = {
    10: {
        "name": "Tree cover",
        "color": "#34D399",
    },
    20: {
        "name": "Shrubland",
        "color": "#A3E635",
    },
    30: {
        "name": "Grassland",
        "color": "#FACC15",
    },
    40: {
        "name": "Cropland",
        "color": "#F59E0B",
    },
    50: {
        "name": "Built-up",
        "color": "#94A3B8",
    },
    60: {
        "name": "Bare / sparse vegetation",
        "color": "#D6D3D1",
    },
    80: {
        "name": "Water",
        "color": "#22D3EE",
    },
    90: {
        "name": "Herbaceous wetland",
        "color": "#2DD4BF",
    },
    95: {
        "name": "Mangroves",
        "color": "#059669",
    },
    100: {
        "name": "Moss and lichen",
        "color": "#C4B5FD",
    },
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Team Kanyarasi Satellite Intelligence API",
    version="1.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://verdant-croissant-7eb837.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ML MODEL
# ============================================================

print("==========================================")
print("Loading ML model...")

clf = joblib.load(MODEL_PATH)

print("ML model loaded successfully.")
print("==========================================")


# ============================================================
# PLANETARY COMPUTER
# ============================================================

print("Connecting to Microsoft Planetary Computer...")

catalog = pystac_client.Client.open(
    CATALOG_URL,
    modifier=planetary_computer.sign_inplace,
)

print("Planetary Computer connected.")


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    lat: float
    lon: float


# ============================================================
# VALIDATION
# ============================================================

def validate_coordinates(
    lat: float,
    lon: float,
):
    if not -90 <= lat <= 90:
        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not -180 <= lon <= 180:
        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )


# ============================================================
# BBOX
# ============================================================

def make_bbox(
    lat: float,
    lon: float,
):
    return [
        lon - BUFFER_DEG,
        lat - BUFFER_DEG,
        lon + BUFFER_DEG,
        lat + BUFFER_DEG,
    ]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "team-kanyarasi-inference",
        "version": "1.2.0",
    }


# ============================================================
# RGB IMAGE PROCESSING
# ============================================================

def enhance_channel(channel):
    """
    Improve visual contrast using percentile stretching.
    """

    channel = channel.astype("float32")

    valid = channel[np.isfinite(channel)]

    if valid.size == 0:
        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    low = np.percentile(
        valid,
        2,
    )

    high = np.percentile(
        valid,
        98,
    )

    if high <= low:
        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    channel = (
        channel - low
    ) / (
        high - low + 1e-6
    )

    return np.clip(
        channel,
        0,
        1,
    )


def create_rgb_image(
    red,
    green,
    blue,
):
    """
    Create enhanced true-color Sentinel-2 imagery.

    B04 = Red
    B03 = Green
    B02 = Blue
    """

    red = enhance_channel(red)
    green = enhance_channel(green)
    blue = enhance_channel(blue)

    gamma = 0.85

    red = np.power(
        red,
        gamma,
    )

    green = np.power(
        green,
        gamma,
    )

    blue = np.power(
        blue,
        gamma,
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb = np.clip(
        rgb * 255,
        0,
        255,
    ).astype("uint8")

    image = Image.fromarray(
        rgb,
        mode="RGB",
    )

    max_dimension = 1600

    if max(image.size) > max_dimension:

        scale = (
            max_dimension
            / max(image.size)
        )

        new_size = (
            max(
                1,
                int(
                    image.size[0]
                    * scale
                ),
            ),
            max(
                1,
                int(
                    image.size[1]
                    * scale
                ),
            ),
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# LOAD SINGLE BAND
# ============================================================

def load_band(
    item,
    band_name,
    bbox,
):
    """
    Download and clip one Sentinel-2 band
    around the selected AOI.
    """

    if band_name not in item.assets:
        raise RuntimeError(
            f"Sentinel-2 scene is missing asset {band_name}"
        )

    da = rioxarray.open_rasterio(
        item.assets[band_name].href
    )

    clipped = da.rio.clip_box(
        *bbox,
        crs="EPSG:4326",
    )

    return clipped.squeeze()


# ============================================================
# CALCULATE NDVI FOR SCENE
# ============================================================

def calculate_scene_ndvi(
    item,
    bbox,
):
    """
    Calculate average NDVI from:
        B04 = Red
        B08 = NIR

    NDVI = (NIR - Red) / (NIR + Red)
    """

    red_da = load_band(
        item,
        "B04",
        bbox,
    )

    nir_da = load_band(
        item,
        "B08",
        bbox,
    )

    red = red_da.values.astype(
        "float32"
    )

    nir = nir_da.values.astype(
        "float32"
    )

    ndvi = (
        (nir - red)
        / (
            nir + red + 1e-6
        )
    )

    ndvi = np.where(
        np.isfinite(ndvi),
        ndvi,
        np.nan,
    )

    valid = ndvi[
        np.isfinite(ndvi)
    ]

    if valid.size == 0:
        raise RuntimeError(
            "No valid NDVI pixels found."
        )

    avg_ndvi = float(
        np.mean(valid)
    )

    return (
        ndvi,
        avg_ndvi,
    )


# ============================================================
# VEGETATION HEALTH
# ============================================================

def classify_vegetation_health(
    avg_ndvi: float,
):
    if avg_ndvi > 0.5:
        return "healthy"

    if avg_ndvi > 0.2:
        return "moderate"

    return "sparse/stressed"


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(req: AnalyzeRequest):

    validate_coordinates(
        req.lat,
        req.lon,
    )

    print("\n==========================================")
    print("NEW ANALYSIS REQUEST")
    print("Latitude:", req.lat)
    print("Longitude:", req.lon)
    print("==========================================")

    bbox = make_bbox(
        req.lat,
        req.lon,
    )

    print(
        "Searching Sentinel-2 imagery..."
    )

    search = catalog.search(
        collections=[
            "sentinel-2-l2a"
        ],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={
            "eo:cloud_cover": {
                "lt": MAX_CLOUD,
            }
        },
        max_items=10,
    )

    items = list(
        search.items()
    )

    if not items:
        raise HTTPException(
            status_code=404,
            detail=(
                "No suitable Sentinel-2 imagery "
                "found for this location."
            ),
        )

    # Newest scene first
    items.sort(
        key=lambda x: (
            x.properties.get(
                "datetime",
                "",
            )
        ),
        reverse=True,
    )

    item = items[0]

    scene_datetime = (
        item.properties.get(
            "datetime"
        )
    )

    cloud_cover = (
        item.properties.get(
            "eo:cloud_cover"
        )
    )

    if cloud_cover is None:
        cloud_cover = 0

    print("Scene selected:")
    print("Scene:", item.id)
    print("Date:", scene_datetime)
    print("Cloud:", cloud_cover)

    try:

        # ------------------------------------------------------
        # LOAD RGB + NIR
        # ------------------------------------------------------

        bands = {}

        for band in [
            "B02",
            "B03",
            "B04",
            "B08",
        ]:

            print(
                "Loading",
                band,
            )

            bands[band] = load_band(
                item,
                band,
                bbox,
            )

        blue = bands[
            "B02"
        ].values.astype(
            "float32"
        )

        green = bands[
            "B03"
        ].values.astype(
            "float32"
        )

        red = bands[
            "B04"
        ].values.astype(
            "float32"
        )

        nir = bands[
            "B08"
        ].values.astype(
            "float32"
        )

        # ------------------------------------------------------
        # NDVI
        # ------------------------------------------------------

        ndvi = (
            (nir - red)
            / (
                nir
                + red
                + 1e-6
            )
        )

        # ------------------------------------------------------
        # ML FEATURES
        # ------------------------------------------------------

        X = np.stack(
            [
                blue,
                green,
                red,
                nir,
                ndvi,
            ],
            axis=-1,
        ).reshape(
            -1,
            5,
        )

        valid = (
            ~np.isnan(X).any(
                axis=1
            )
        )

        # ------------------------------------------------------
        # ML PREDICTION
        # ------------------------------------------------------

        print(
            "Running Random Forest..."
        )

        preds = clf.predict(
            X[valid]
        )

        # ------------------------------------------------------
        # LAND COVER
        # ------------------------------------------------------

        unique, counts = np.unique(
            preds,
            return_counts=True,
        )

        total = counts.sum()

        land_cover = []

        for cls, count in zip(
            unique,
            counts,
        ):

            cls_int = int(cls)

            if (
                cls_int
                in WORLDCOVER_CLASSES
            ):

                info = (
                    WORLDCOVER_CLASSES[
                        cls_int
                    ]
                )

                land_cover.append(
                    {
                        "name":
                            info["name"],

                        "value":
                            round(
                                (
                                    float(
                                        count
                                    )
                                    / total
                                    * 100
                                ),
                                1,
                            ),

                        "color":
                            info["color"],
                    }
                )

        land_cover.sort(
            key=lambda x: x[
                "value"
            ],
            reverse=True,
        )

        # ------------------------------------------------------
        # NDVI HEALTH
        # ------------------------------------------------------

        avg_ndvi = float(
            np.nanmean(ndvi)
        )

        vegetation_health = (
            classify_vegetation_health(
                avg_ndvi
            )
        )

        # ------------------------------------------------------
        # TRUE COLOR IMAGE
        # ------------------------------------------------------

        print(
            "Creating satellite preview..."
        )

        satellite_image = (
            create_rgb_image(
                red,
                green,
                blue,
            )
        )

        # ------------------------------------------------------
        # RESPONSE
        # ------------------------------------------------------

        response = {

            "coordinates": {
                "lat": req.lat,
                "lon": req.lon,
            },

            "scene": {
                "id": item.id,

                "date":
                    scene_datetime,

                "cloud_cover":
                    float(
                        cloud_cover
                    ),
            },

            "satellite_image":
                satellite_image,

            "land_cover":
                land_cover,

            "avg_ndvi":
                round(
                    avg_ndvi,
                    3,
                ),

            "vegetation_health":
                vegetation_health,
        }

        print(
            "Analysis completed successfully."
        )

        return response

    except Exception as e:

        print(
            "ANALYSIS ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed: "
                + str(e)
            ),
        )


# ============================================================
# NDVI HISTORY
# ============================================================

@app.get("/ndvi-history")
def ndvi_history(
    lat: float,
    lon: float,
):

    validate_coordinates(
        lat,
        lon,
    )

    print("\n==========================================")
    print("NDVI HISTORY REQUEST")
    print("Latitude:", lat)
    print("Longitude:", lon)
    print("==========================================")

    bbox = make_bbox(
        lat,
        lon,
    )

    # --------------------------------------------------------
    # DATE WINDOW
    # --------------------------------------------------------

    end_date = date.today()

    start_date = (
        end_date
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    history_range = (
        f"{start_date.isoformat()}/"
        f"{end_date.isoformat()}"
    )

    print(
        "Historical range:",
        history_range,
    )

    try:

        # ----------------------------------------------------
        # SEARCH HISTORICAL SCENES
        # ----------------------------------------------------

        search = catalog.search(
            collections=[
                "sentinel-2-l2a"
            ],
            bbox=bbox,
            datetime=history_range,
            query={
                "eo:cloud_cover": {
                    "lt": MAX_CLOUD,
                }
            },
            max_items=100,
        )

        items = list(
            search.items()
        )

        if not items:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No historical Sentinel-2 "
                    "observations found for this location."
                ),
            )

        # ----------------------------------------------------
        # GROUP BY MONTH
        #
        # We choose one scene per month.
        # Within each month, choose the scene
        # with the lowest cloud cover.
        # ----------------------------------------------------

        monthly_candidates = {}

        for item in items:

            scene_datetime = (
                item.properties.get(
                    "datetime"
                )
            )

            if not scene_datetime:
                continue

            scene_date = (
                scene_datetime[:10]
            )

            month_key = (
                scene_date[:7]
            )

            cloud = (
                item.properties.get(
                    "eo:cloud_cover"
                )
            )

            if cloud is None:
                cloud = 100

            current = (
                monthly_candidates.get(
                    month_key
                )
            )

            if (
                current is None
                or cloud
                < current["cloud_cover"]
            ):

                monthly_candidates[
                    month_key
                ] = {
                    "item": item,
                    "cloud_cover": float(
                        cloud
                    ),
                    "date": scene_datetime,
                }

        # ----------------------------------------------------
        # TAKE MOST RECENT MONTHS
        # ----------------------------------------------------

        selected_months = sorted(
            monthly_candidates.keys(),
            reverse=True,
        )[
            :HISTORY_POINTS
        ]

        selected_months.sort()

        print(
            "Selected months:",
            selected_months,
        )

        # ----------------------------------------------------
        # CALCULATE NDVI
        # ----------------------------------------------------

        observations = []

        for month in selected_months:

            candidate = (
                monthly_candidates[
                    month
                ]
            )

            item = candidate[
                "item"
            ]

            print(
                "Processing historical scene:",
                item.id,
                candidate["date"],
            )

            try:

                _ndvi, avg_ndvi = (
                    calculate_scene_ndvi(
                        item,
                        bbox,
                    )
                )

                observations.append(
                    {
                        "date":
                            candidate[
                                "date"
                            ],

                        "period":
                            month,

                        "ndvi":
                            round(
                                avg_ndvi,
                                3,
                            ),

                        "cloud_cover":
                            round(
                                candidate[
                                    "cloud_cover"
                                ],
                                1,
                            ),

                        "scene_id":
                            item.id,
                    }
                )

            except Exception as scene_error:

                print(
                    "Skipping historical scene:",
                    item.id,
                    str(scene_error),
                )

                continue

        if not observations:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Historical scenes were found, "
                    "but valid NDVI could not be calculated."
                ),
            )

        # ----------------------------------------------------
        # SORT CHRONOLOGICALLY
        # ----------------------------------------------------

        observations.sort(
            key=lambda x: x["date"]
        )

        # ----------------------------------------------------
        # TREND SUMMARY
        # ----------------------------------------------------

        first_ndvi = (
            observations[0]["ndvi"]
        )

        latest_ndvi = (
            observations[-1]["ndvi"]
        )

        change = (
            latest_ndvi
            - first_ndvi
        )

        if change > 0.05:
            trend = "increasing"

        elif change < -0.05:
            trend = "decreasing"

        else:
            trend = "stable"

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "coordinates": {
                "lat": lat,
                "lon": lon,
            },

            "period": {
                "start":
                    observations[
                        0
                    ]["date"],

                "end":
                    observations[
                        -1
                    ]["date"],
            },

            "observations":
                observations,

            "summary": {

                "first_ndvi":
                    round(
                        first_ndvi,
                        3,
                    ),

                "latest_ndvi":
                    round(
                        latest_ndvi,
                        3,
                    ),

                "change":
                    round(
                        change,
                        3,
                    ),

                "trend":
                    trend,

                "observation_count":
                    len(
                        observations
                    ),
            },
        }

        print(
            "NDVI history completed:",
            len(observations),
            "observations",
        )

        return response

    except HTTPException:
        raise

    except Exception as e:

        print(
            "NDVI HISTORY ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "NDVI history failed: "
                + str(e)
            ),
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service":
            "Team Kanyarasi Satellite Intelligence API",

        "status":
            "running",

        "endpoints": [
            "/health",
            "/analyze",
            "/ndvi-history",
        ],
    }