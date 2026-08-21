"""
Team Kanyarasi — Sentinel-2 Farm Analysis API

Endpoints:

GET  /health
POST /analyze
GET  /ndvi-history?lat=<lat>&lon=<lon>

Architecture:

/analyze
    -> latest Sentinel-2 scene
    -> NDVI
    -> vegetation health
    -> compact spatial NDVI/stress map
    -> Random Forest land-cover classification
    -> field condition
    -> fast crop-stress early warning
    -> true-color preview

/ndvi-history
    -> historical Sentinel-2 search
    -> one suitable observation per month
    -> real NDVI calculation
    -> temporal trend

IMPORTANT:
Historical processing is intentionally NOT performed inside
/analyze so the main analysis remains responsive on small
cloud instances.
"""

import base64
import gc
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

DATE_RANGE = (
    f"2025-08-01/{date.today().isoformat()}"
)

HISTORY_DAYS = 365

HISTORY_POINTS = 12

MAX_CLOUD = 30

BUFFER_DEG = 0.03

# ------------------------------------------------------------
# Spatial map
# ------------------------------------------------------------

# 8 x 8 = 64 cells.
# Enough visual detail for the UI while keeping the response
# small and fast.
SPATIAL_GRID_SIZE = 8

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

# Process model features in chunks.
ML_CHUNK_SIZE = 20000

# Do not classify every single raster pixel.
#
# A stride of 2 means:
#   every second row
#   every second column
#
# This dramatically reduces Random Forest CPU usage while
# preserving a representative land-cover estimate.
ML_PIXEL_STRIDE = 2

# ------------------------------------------------------------
# Image
# ------------------------------------------------------------

IMAGE_MAX_DIMENSION = 768

IMAGE_JPEG_QUALITY = 78


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
    version="1.8.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://verdant-croissant-7eb837.netlify.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
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
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "team-kanyarasi-inference",
        "version": "1.8.0",
        "memory_optimized": True,
        "spatial_grid": f"{SPATIAL_GRID_SIZE}x{SPATIAL_GRID_SIZE}",
        "historical_processing_in_analyze": False,
    }


# ============================================================
# IMAGE PROCESSING
# ============================================================

def enhance_channel(
    channel: np.ndarray,
):
    channel = np.asarray(
        channel,
        dtype=np.float32,
    )

    valid = channel[
        np.isfinite(channel)
    ]

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

    normalized = (
        channel - low
    ) / (
        high - low + 1e-6
    )

    return np.clip(
        normalized,
        0,
        1,
    )


def create_rgb_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
):
    """
    Create compact true-color Sentinel-2 preview.

    B04 = Red
    B03 = Green
    B02 = Blue
    """

    height, width = red.shape

    largest_dimension = max(
        height,
        width,
    )

    if (
        largest_dimension
        > IMAGE_MAX_DIMENSION
    ):
        step = int(
            np.ceil(
                largest_dimension
                / IMAGE_MAX_DIMENSION
            )
        )
    else:
        step = 1

    red_small = red[
        ::step,
        ::step,
    ]

    green_small = green[
        ::step,
        ::step,
    ]

    blue_small = blue[
        ::step,
        ::step,
    ]

    red_small = enhance_channel(
        red_small
    )

    green_small = enhance_channel(
        green_small
    )

    blue_small = enhance_channel(
        blue_small
    )

    gamma = 0.85

    red_small = np.power(
        red_small,
        gamma,
    )

    green_small = np.power(
        green_small,
        gamma,
    )

    blue_small = np.power(
        blue_small,
        gamma,
    )

    rgb = np.stack(
        [
            red_small,
            green_small,
            blue_small,
        ],
        axis=-1,
    )

    rgb = np.clip(
        rgb * 255,
        0,
        255,
    ).astype(
        "uint8"
    )

    image = Image.fromarray(
        rgb,
        mode="RGB",
    )

    if (
        max(image.size)
        > IMAGE_MAX_DIMENSION
    ):

        scale = (
            IMAGE_MAX_DIMENSION
            / max(image.size)
        )

        image = image.resize(
            (
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
            ),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=IMAGE_JPEG_QUALITY,
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
# LOAD BAND
# ============================================================

def load_band(
    item,
    band_name,
    bbox,
):
    """
    Load a single Sentinel-2 band and immediately return
    its NumPy representation.

    Planetary Computer assets are signed automatically through
    planetary_computer.sign_inplace.
    """

    if band_name not in item.assets:
        raise RuntimeError(
            f"Sentinel-2 scene is missing asset {band_name}"
        )

    print(
        "Loading",
        band_name,
    )

    raster = rioxarray.open_rasterio(
        item.assets[
            band_name
        ].href
    )

    clipped = None

    try:

        clipped = raster.rio.clip_box(
            *bbox,
            crs="EPSG:4326",
        )

        values = clipped.squeeze().values

        return np.asarray(
            values,
            dtype=np.float32,
        )

    finally:

        del raster

        if clipped is not None:
            del clipped

        gc.collect()


# ============================================================
# NDVI
# ============================================================

def calculate_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
):
    denominator = (
        nir
        + red
        + 1e-6
    )

    ndvi = (
        nir - red
    ) / denominator

    ndvi = np.where(
        np.isfinite(ndvi),
        ndvi,
        np.nan,
    )

    return ndvi.astype(
        np.float32,
        copy=False,
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
# SPATIAL CELL CLASSIFICATION
# ============================================================

def classify_spatial_cell(
    ndvi_value: float,
):
    if not np.isfinite(
        ndvi_value
    ):

        return {
            "category": "no-data",
            "stress": None,
        }

    if ndvi_value >= 0.5:

        return {
            "category": "healthy",
            "stress": round(
                max(
                    0,
                    min(
                        100,
                        100
                        - ndvi_value * 100,
                    ),
                )
            ),
        }

    if ndvi_value >= 0.3:

        stress = (
            (
                0.5
                - ndvi_value
            )
            / 0.5
        ) * 60 + 20

        return {
            "category": "moderate",
            "stress": round(
                max(
                    0,
                    min(
                        100,
                        stress,
                    ),
                )
            ),
        }

    if ndvi_value >= 0.15:

        stress = (
            (
                0.3
                - ndvi_value
            )
            / 0.3
        ) * 60 + 40

        return {
            "category": "stressed",
            "stress": round(
                max(
                    0,
                    min(
                        100,
                        stress,
                    ),
                )
            ),
        }

    stress = (
        75
        + (
            max(
                0,
                0.15
                - ndvi_value,
            )
            / 0.15
        ) * 25
    )

    return {
        "category": "high-stress",
        "stress": round(
            max(
                0,
                min(
                    100,
                    stress,
                ),
            )
        ),
    }


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

def calculate_spatial_analysis(
    ndvi: np.ndarray,
    bbox,
):
    """
    Create a compact 8x8 spatial NDVI/stress grid from the
    actual Sentinel-2 NDVI raster.

    No extra satellite request is required.
    """

    valid = ndvi[
        np.isfinite(ndvi)
    ]

    if valid.size == 0:

        raise RuntimeError(
            "No valid NDVI pixels available."
        )

    rows, cols = ndvi.shape

    target_rows = min(
        SPATIAL_GRID_SIZE,
        rows,
    )

    target_cols = min(
        SPATIAL_GRID_SIZE,
        cols,
    )

    min_lon, min_lat, max_lon, max_lat = bbox

    lon_edges = np.linspace(
        min_lon,
        max_lon,
        target_cols + 1,
    )

    lat_edges = np.linspace(
        min_lat,
        max_lat,
        target_rows + 1,
    )

    cells = []

    counts = {
        "healthy": 0,
        "moderate": 0,
        "stressed": 0,
        "high-stress": 0,
        "no-data": 0,
    }

    for row in range(
        target_rows
    ):

        row_start = int(
            row
            * rows
            / target_rows
        )

        row_end = int(
            (row + 1)
            * rows
            / target_rows
        )

        row_end = max(
            row_end,
            row_start + 1,
        )

        for col in range(
            target_cols
        ):

            col_start = int(
                col
                * cols
                / target_cols
            )

            col_end = int(
                (col + 1)
                * cols
                / target_cols
            )

            col_end = max(
                col_end,
                col_start + 1,
            )

            block = ndvi[
                row_start:row_end,
                col_start:col_end,
            ]

            block_valid = block[
                np.isfinite(block)
            ]

            bounds = {
                "south": float(
                    lat_edges[row]
                ),
                "north": float(
                    lat_edges[row + 1]
                ),
                "west": float(
                    lon_edges[col]
                ),
                "east": float(
                    lon_edges[col + 1]
                ),
            }

            if block_valid.size == 0:

                counts[
                    "no-data"
                ] += 1

                cells.append(
                    {
                        "row": row,
                        "col": col,
                        "ndvi": None,
                        "stress": None,
                        "category": "no-data",
                        "bounds": bounds,
                    }
                )

                continue

            cell_ndvi = float(
                np.mean(
                    block_valid
                )
            )

            cell_info = (
                classify_spatial_cell(
                    cell_ndvi
                )
            )

            category = cell_info[
                "category"
            ]

            counts[
                category
            ] += 1

            cells.append(
                {
                    "row": row,
                    "col": col,
                    "ndvi": round(
                        cell_ndvi,
                        3,
                    ),
                    "stress": cell_info[
                        "stress"
                    ],
                    "category": category,
                    "bounds": bounds,
                }
            )

    valid_cells = (
        counts["healthy"]
        + counts["moderate"]
        + counts["stressed"]
        + counts["high-stress"]
    )

    if valid_cells == 0:

        raise RuntimeError(
            "No valid spatial cells generated."
        )

    return {
        "grid_size": {
            "rows": target_rows,
            "cols": target_cols,
        },

        "resolution": (
            "Downsampled NDVI grid derived "
            "from Sentinel-2 B04/B08"
        ),

        "bounds": {
            "south": float(
                min_lat
            ),
            "north": float(
                max_lat
            ),
            "west": float(
                min_lon
            ),
            "east": float(
                max_lon
            ),
        },

        "cells": cells,

        "summary": {
            "healthy_percent": round(
                counts["healthy"]
                / valid_cells
                * 100,
                1,
            ),
            "moderate_percent": round(
                counts["moderate"]
                / valid_cells
                * 100,
                1,
            ),
            "stressed_percent": round(
                counts["stressed"]
                / valid_cells
                * 100,
                1,
            ),
            "high_stress_percent": round(
                counts["high-stress"]
                / valid_cells
                * 100,
                1,
            ),
            "elevated_stress_percent": round(
                (
                    counts["stressed"]
                    + counts["high-stress"]
                )
                / valid_cells
                * 100,
                1,
            ),
            "valid_cells": valid_cells,
            "no_data_cells": counts[
                "no-data"
            ],
        },
    }


# ============================================================
# FIELD CONDITION
# ============================================================

def calculate_soil_condition(
    avg_ndvi: float,
    ndvi: np.ndarray,
    land_cover: list,
):
    """
    Satellite-derived field-condition indicator.

    This is NOT a laboratory soil measurement.
    """

    vegetation_signal = round(
        max(
            0,
            min(
                100,
                avg_ndvi * 100,
            ),
        )
    )

    bare_percentage = 0.0

    for item in land_cover:

        if (
            item["name"]
            == "Bare / sparse vegetation"
        ):

            bare_percentage = float(
                item["value"]
            )

            break

    bare_soil_condition = round(
        max(
            0,
            min(
                100,
                100
                - bare_percentage * 2,
            ),
        )
    )

    valid_ndvi = ndvi[
        np.isfinite(ndvi)
    ]

    if valid_ndvi.size == 0:

        consistency = 50

    else:

        ndvi_std = float(
            np.std(
                valid_ndvi
            )
        )

        consistency = round(
            max(
                0,
                min(
                    100,
                    100
                    - ndvi_std * 250,
                ),
            )
        )

    score = round(
        vegetation_signal * 0.50
        + bare_soil_condition * 0.20
        + consistency * 0.30
    )

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    if score >= 70:
        status = "good"

    elif score >= 40:
        status = "moderate"

    else:
        status = "stressed"

    return {
        "score": score,
        "status": status,
        "vegetation_signal": vegetation_signal,
        "bare_sparse_cover": round(
            bare_percentage,
            1,
        ),
        "spatial_consistency": consistency,
    }


# ============================================================
# FAST STRESS RISK
# ============================================================

def calculate_fast_stress_risk(
    avg_ndvi: float,
    soil_condition: dict,
    cloud_cover: float,
):
    """
    Fast current-scene stress indicator.

    Historical trend is deliberately excluded from /analyze.
    The frontend can combine this with /ndvi-history.
    """

    vegetation_stress = round(
        max(
            0,
            min(
                100,
                (
                    (
                        0.7
                        - avg_ndvi
                    )
                    / 0.7
                )
                * 100,
            ),
        )
    )

    spatial_consistency = int(
        soil_condition.get(
            "spatial_consistency",
            50,
        )
    )

    spatial_stress = round(
        max(
            0,
            min(
                100,
                100
                - spatial_consistency,
            ),
        )
    )

    field_condition_score = int(
        soil_condition.get(
            "score",
            50,
        )
    )

    field_condition_stress = round(
        max(
            0,
            min(
                100,
                100
                - field_condition_score,
            ),
        )
    )

    bare_sparse_cover = float(
        soil_condition.get(
            "bare_sparse_cover",
            0,
        )
    )

    bare_cover_stress = round(
        max(
            0,
            min(
                100,
                bare_sparse_cover * 2,
            ),
        )
    )

    observation_quality = round(
        max(
            0,
            min(
                100,
                100 - cloud_cover,
            ),
        )
    )

    risk_score = round(
        vegetation_stress * 0.45
        + field_condition_stress * 0.25
        + spatial_stress * 0.15
        + bare_cover_stress * 0.15
    )

    risk_score = max(
        0,
        min(
            100,
            risk_score,
        ),
    )

    if risk_score >= 70:

        risk_level = "high"

    elif risk_score >= 40:

        risk_level = "moderate"

    else:

        risk_level = "low"

    drivers = []

    if vegetation_stress >= 60:

        drivers.append(
            "low vegetation response"
        )

    if field_condition_stress >= 50:

        drivers.append(
            "reduced field condition"
        )

    if spatial_stress >= 30:

        drivers.append(
            "spatial vegetation variability"
        )

    if bare_cover_stress >= 25:

        drivers.append(
            "increased bare/sparse cover"
        )

    if not drivers:

        drivers.append(
            "no dominant stress signal"
        )

    if vegetation_stress >= 60:

        primary_signal = (
            "low vegetation response"
        )

    elif field_condition_stress >= 50:

        primary_signal = (
            "reduced field condition"
        )

    elif spatial_stress >= 30:

        primary_signal = (
            "spatial vegetation variability"
        )

    else:

        primary_signal = (
            "no dominant stress signal"
        )

    if risk_level == "high":

        interpretation = (
            "Elevated crop-stress signal detected. "
            "Field inspection and follow-up satellite "
            "observation are recommended."
        )

    elif risk_level == "moderate":

        interpretation = (
            "Moderate crop-stress signal detected. "
            "Continue monitoring and inspect areas showing "
            "vegetation deterioration."
        )

    else:

        interpretation = (
            "No strong crop-stress signal detected "
            "in the current satellite observation."
        )

    confidence = observation_quality

    return {
        "score": risk_score,
        "level": risk_level,
        "confidence": confidence,
        "primary_signal": primary_signal,
        "drivers": drivers,
        "interpretation": interpretation,
        "factors": {
            "vegetation_stress":
                vegetation_stress,
            "spatial_stress":
                spatial_stress,
            "field_condition_stress":
                field_condition_stress,
            "bare_cover_stress":
                bare_cover_stress,
            "observation_quality":
                observation_quality,
        },
        "temporal_context": {
            "historical_change": None,
            "historical_trend": "available via /ndvi-history",
        },
        "disclaimer": (
            "This is a satellite-derived crop-stress "
            "and pest-risk early-warning indicator. "
            "It does not identify a specific pest or "
            "disease and should be validated with "
            "field observations."
        ),
    }


# ============================================================
# LAND COVER
# ============================================================

def classify_land_cover(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    ndvi: np.ndarray,
):
    """
    Random Forest land-cover classification.

    To reduce CPU/memory usage, every ML_PIXEL_STRIDE-th
    pixel is sampled before prediction.
    """

    # --------------------------------------------------------
    # Downsample the prediction input
    # --------------------------------------------------------

    blue_sample = blue[
        ::ML_PIXEL_STRIDE,
        ::ML_PIXEL_STRIDE,
    ]

    green_sample = green[
        ::ML_PIXEL_STRIDE,
        ::ML_PIXEL_STRIDE,
    ]

    red_sample = red[
        ::ML_PIXEL_STRIDE,
        ::ML_PIXEL_STRIDE,
    ]

    nir_sample = nir[
        ::ML_PIXEL_STRIDE,
        ::ML_PIXEL_STRIDE,
    ]

    ndvi_sample = ndvi[
        ::ML_PIXEL_STRIDE,
        ::ML_PIXEL_STRIDE,
    ]

    blue_flat = blue_sample.reshape(-1)
    green_flat = green_sample.reshape(-1)
    red_flat = red_sample.reshape(-1)
    nir_flat = nir_sample.reshape(-1)
    ndvi_flat = ndvi_sample.reshape(-1)

    del blue_sample
    del green_sample
    del red_sample
    del nir_sample
    del ndvi_sample

    class_counts = {}

    total_pixels = len(
        ndvi_flat
    )

    for start in range(
        0,
        total_pixels,
        ML_CHUNK_SIZE,
    ):

        end = min(
            start + ML_CHUNK_SIZE,
            total_pixels,
        )

        b = blue_flat[
            start:end
        ]

        g = green_flat[
            start:end
        ]

        r = red_flat[
            start:end
        ]

        n = nir_flat[
            start:end
        ]

        v = ndvi_flat[
            start:end
        ]

        valid_mask = (
            np.isfinite(b)
            & np.isfinite(g)
            & np.isfinite(r)
            & np.isfinite(n)
            & np.isfinite(v)
        )

        if not np.any(
            valid_mask
        ):
            continue

        X_chunk = np.column_stack(
            [
                b[valid_mask],
                g[valid_mask],
                r[valid_mask],
                n[valid_mask],
                v[valid_mask],
            ]
        )

        predictions = clf.predict(
            X_chunk
        )

        unique, counts = np.unique(
            predictions,
            return_counts=True,
        )

        for cls, count in zip(
            unique,
            counts,
        ):

            cls_int = int(
                cls
            )

            class_counts[
                cls_int
            ] = (
                class_counts.get(
                    cls_int,
                    0,
                )
                + int(count)
            )

        del X_chunk
        del predictions

        gc.collect()

    del blue_flat
    del green_flat
    del red_flat
    del nir_flat
    del ndvi_flat

    gc.collect()

    total_predictions = sum(
        class_counts.values()
    )

    land_cover = []

    if total_predictions <= 0:

        return land_cover

    for (
        cls_int,
        count,
    ) in class_counts.items():

        info = WORLDCOVER_CLASSES.get(
            cls_int
        )

        if info is None:
            continue

        percentage = (
            count
            / total_predictions
            * 100
        )

        land_cover.append(
            {
                "name":
                    info["name"],

                "value":
                    round(
                        percentage,
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

    return land_cover


# ============================================================
# HISTORICAL NDVI
# ============================================================

def get_ndvi_history_data(
    lat: float,
    lon: float,
):
    """
    Historical processing lives here instead of /analyze.
    """

    bbox = make_bbox(
        lat,
        lon,
    )

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
        return None

    monthly_candidates = {}

    for item in items:

        scene_datetime = (
            item.properties.get(
                "datetime"
            )
        )

        if not scene_datetime:
            continue

        month_key = (
            scene_datetime[:7]
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

    selected_months = sorted(
        monthly_candidates.keys(),
        reverse=True,
    )[:HISTORY_POINTS]

    selected_months.sort()

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

            red = load_band(
                item,
                "B04",
                bbox,
            )

            nir = load_band(
                item,
                "B08",
                bbox,
            )

            ndvi = calculate_ndvi(
                red,
                nir,
            )

            valid = ndvi[
                np.isfinite(ndvi)
            ]

            if valid.size == 0:
                continue

            avg_ndvi = float(
                np.mean(valid)
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

            del red
            del nir
            del ndvi

            gc.collect()

        except Exception as scene_error:

            print(
                "Skipping historical scene:",
                item.id,
                str(
                    scene_error
                ),
            )

        gc.collect()

    if not observations:
        return None

    observations.sort(
        key=lambda x: x[
            "date"
        ]
    )

    first_ndvi = (
        observations[
            0
        ]["ndvi"]
    )

    latest_ndvi = (
        observations[
            -1
        ]["ndvi"]
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

    return {
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


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(
    req: AnalyzeRequest,
):

    validate_coordinates(
        req.lat,
        req.lon,
    )

    print("\n==========================================")
    print("NEW FAST ANALYSIS REQUEST")
    print("Latitude:", req.lat)
    print("Longitude:", req.lon)
    print("==========================================")

    bbox = make_bbox(
        req.lat,
        req.lon,
    )

    # ========================================================
    # FIND LATEST SCENE
    # ========================================================

    print(
        "Searching latest Sentinel-2 scene..."
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

    print(
        "Scene selected:",
        item.id,
    )

    print(
        "Date:",
        scene_datetime,
    )

    print(
        "Cloud:",
        cloud_cover,
    )

    try:

        # ====================================================
        # LOAD RED + NIR
        # ====================================================

        red = load_band(
            item,
            "B04",
            bbox,
        )

        nir = load_band(
            item,
            "B08",
            bbox,
        )

        print(
            "Calculating NDVI..."
        )

        ndvi = calculate_ndvi(
            red,
            nir,
        )

        valid_ndvi = ndvi[
            np.isfinite(ndvi)
        ]

        if valid_ndvi.size == 0:

            raise RuntimeError(
                "No valid NDVI pixels found."
            )

        avg_ndvi = float(
            np.mean(
                valid_ndvi
            )
        )

        vegetation_health = (
            classify_vegetation_health(
                avg_ndvi
            )
        )

        print(
            "Average NDVI:",
            round(
                avg_ndvi,
                3,
            ),
        )

        # ====================================================
        # SPATIAL MAP
        # ====================================================

        print(
            "Calculating spatial NDVI/stress map..."
        )

        spatial_analysis = (
            calculate_spatial_analysis(
                ndvi,
                bbox,
            )
        )

        print(
            "Spatial summary:",
            spatial_analysis[
                "summary"
            ],
        )

        # ====================================================
        # LOAD RGB
        # ====================================================

        blue = load_band(
            item,
            "B02",
            bbox,
        )

        green = load_band(
            item,
            "B03",
            bbox,
        )

        # ====================================================
        # LAND COVER
        # ========================================================

        print(
            "Running sampled Random Forest..."
        )

        land_cover = classify_land_cover(
            blue,
            green,
            red,
            nir,
            ndvi,
        )

        print(
            "Land cover:",
            land_cover,
        )

        # ====================================================
        # FIELD CONDITION
        # ====================================================

        print(
            "Calculating satellite-derived field condition..."
        )

        soil_condition = (
            calculate_soil_condition(
                avg_ndvi=avg_ndvi,
                ndvi=ndvi,
                land_cover=land_cover,
            )
        )

        print(
            "Field condition:",
            soil_condition,
        )

        # ====================================================
        # FAST STRESS RISK
        # ====================================================

        print(
            "Calculating current stress risk..."
        )

        stress_risk = (
            calculate_fast_stress_risk(
                avg_ndvi=avg_ndvi,
                soil_condition=soil_condition,
                cloud_cover=float(
                    cloud_cover
                ),
            )
        )

        print(
            "Stress risk:",
            stress_risk,
        )

        # ====================================================
        # TRUE COLOR IMAGE
        # ====================================================

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

        # ====================================================
        # TEMPORARY MEMORY CLEANUP
        # ====================================================

        del blue
        del green
        del red
        del nir
        del ndvi
        del valid_ndvi

        gc.collect()

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {
            "coordinates": {
                "lat": req.lat,
                "lon": req.lon,
            },

            "scene": {
                "id": item.id,
                "date": scene_datetime,
                "cloud_cover": float(
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

            "soil_condition":
                soil_condition,

            "stress_risk":
                stress_risk,

            "spatial_analysis":
                spatial_analysis,

            "temporal_summary": {
                "first_ndvi": None,
                "latest_ndvi": None,
                "change": None,
                "trend": "available via /ndvi-history",
                "observation_count": 0,
            },

            "analysis_mode":
                "fast-current-scene",

            "history_endpoint":
                "/ndvi-history",
        }

        print(
            "Fast analysis completed successfully."
        )

        return response

    except Exception as e:

        print(
            "ANALYSIS ERROR:",
            str(e),
        )

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed: "
                + str(e)
            ),
        )


# ============================================================
# NDVI HISTORY ENDPOINT
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

    try:

        history_data = (
            get_ndvi_history_data(
                lat,
                lon,
            )
        )

        if not history_data:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Historical scenes were found, "
                    "but valid NDVI could not be calculated."
                ),
            )

        observations = (
            history_data[
                "observations"
            ]
        )

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

            "summary":
                history_data[
                    "summary"
                ],
        }

        gc.collect()

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

        gc.collect()

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

        "version":
            "1.8.0",

        "memory_optimized":
            True,

        "fast_analysis":
            True,

        "historical_processing_in_analyze":
            False,

        "spatial_grid":
            f"{SPATIAL_GRID_SIZE}x{SPATIAL_GRID_SIZE}",

        "ml_pixel_stride":
            ML_PIXEL_STRIDE,

        "endpoints": [
            "/health",
            "/analyze",
            "/ndvi-history",
        ],
    }