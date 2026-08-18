"""
On-demand Sentinel-2 farm analysis API.

Input:
    latitude + longitude

Output:
    - latest suitable Sentinel-2 scene
    - true-color satellite image
    - cloud cover
    - NDVI
    - vegetation health
    - ML land-cover classification
"""

import io
import base64

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

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Search period
DATE_RANGE = "2025-08-01/2026-08-18"

# Maximum acceptable cloud cover
MAX_CLOUD = 30

# Approximately 3 km around requested point
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
    version="1.1.0",
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
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "team-kanyarasi-inference",
    }


# ============================================================
# RGB IMAGE PROCESSING
# ============================================================

def enhance_channel(channel):
    """
    Improve visual contrast using robust percentile stretching.
    """

    channel = channel.astype("float32")

    valid = channel[np.isfinite(channel)]

    if valid.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    channel = (channel - low) / (
        high - low + 1e-6
    )

    channel = np.clip(
        channel,
        0,
        1,
    )

    return channel


def create_rgb_image(red, green, blue):
    """
    Create enhanced true-color Sentinel-2 imagery.

    Sentinel-2:
        B04 = Red
        B03 = Green
        B02 = Blue
    """

    red = enhance_channel(red)
    green = enhance_channel(green)
    blue = enhance_channel(blue)

    # Gamma correction
    gamma = 0.85

    red = np.power(red, gamma)
    green = np.power(green, gamma)
    blue = np.power(blue, gamma)

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

    # Keep response size reasonable
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
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(req: AnalyzeRequest):

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    if not -90 <= req.lat <= 90:
        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not -180 <= req.lon <= 180:
        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )

    print("\n==========================================")
    print("NEW ANALYSIS REQUEST")
    print("Latitude:", req.lat)
    print("Longitude:", req.lon)
    print("==========================================")


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox = [
        req.lon - BUFFER_DEG,
        req.lat - BUFFER_DEG,
        req.lon + BUFFER_DEG,
        req.lat + BUFFER_DEG,
    ]


    # --------------------------------------------------------
    # Search Sentinel-2
    # --------------------------------------------------------

    print("Searching Sentinel-2 imagery...")

    search = catalog.search(
        collections=[
            "sentinel-2-l2a"
        ],

        bbox=bbox,

        datetime=DATE_RANGE,

        query={
            "eo:cloud_cover": {
                "lt": MAX_CLOUD
            }
        },

        # Get several candidates so we can select
        # the newest suitable scene.
        max_items=10,
    )

    items = list(search.items())

    if not items:
        raise HTTPException(
            status_code=404,
            detail=(
                "No suitable Sentinel-2 imagery "
                "found for this location."
            ),
        )


    # --------------------------------------------------------
    # Select latest scene
    # --------------------------------------------------------

    items.sort(
        key=lambda x: (
            x.properties.get(
                "datetime",
                ""
            )
        ),
        reverse=True,
    )

    item = items[0]

    scene_datetime = item.properties.get(
        "datetime"
    )

    cloud_cover = item.properties.get(
        "eo:cloud_cover"
    )

    if cloud_cover is None:
        cloud_cover = 0


    print("Scene selected:")
    print("Scene:", item.id)
    print("Date:", scene_datetime)
    print(
        "Cloud:",
        cloud_cover,
    )


    try:

        # ----------------------------------------------------
        # LOAD BANDS
        # ----------------------------------------------------

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

            da = rioxarray.open_rasterio(
                item.assets[band].href
            )

            clipped = da.rio.clip_box(
                *bbox,
                crs="EPSG:4326",
            )

            bands[band] = clipped.squeeze()


        # ----------------------------------------------------
        # NUMPY ARRAYS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        ndvi = (
            (nir - red)
            /
            (
                nir
                + red
                + 1e-6
            )
        )


        # ----------------------------------------------------
        # ML FEATURES
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        print(
            "Running Random Forest..."
        )

        preds = clf.predict(
            X[valid]
        )


        # ----------------------------------------------------
        # LAND COVER
        # ----------------------------------------------------

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
                            info[
                                "name"
                            ],

                        "value":
                            round(
                                float(
                                    count
                                )
                                / total
                                * 100,
                                1,
                            ),

                        "color":
                            info[
                                "color"
                            ],
                    }
                )


        land_cover.sort(
            key=lambda x: x[
                "value"
            ],
            reverse=True,
        )


        # ----------------------------------------------------
        # NDVI HEALTH
        # ----------------------------------------------------

        avg_ndvi = float(
            np.nanmean(ndvi)
        )

        if avg_ndvi > 0.5:

            vegetation_health = (
                "healthy"
            )

        elif avg_ndvi > 0.2:

            vegetation_health = (
                "moderate"
            )

        else:

            vegetation_health = (
                "sparse/stressed"
            )


        # ----------------------------------------------------
        # TRUE-COLOR IMAGE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "coordinates": {
                "lat": req.lat,
                "lon": req.lon,
            },

            "scene": {
                "id": item.id,

                "date": scene_datetime,

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