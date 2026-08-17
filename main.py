"""
On-demand land cover / farm analysis inference API.

Takes a lat/lon coordinate, pulls a small window of live Sentinel-2
imagery around it, runs the already-trained Random Forest model on
every pixel, and returns a land cover breakdown + basic vegetation
health stats — computed fresh for that exact location, not precomputed.
"""

import joblib
import numpy as np
import planetary_computer
import pystac_client
import rioxarray
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = "model_india_small.pkl"
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATE_RANGE = "2025-08-01/2026-02-28"
MAX_CLOUD = 25
BUFFER_DEG = 0.03  # ~3km box around the point — small enough to be fast

WORLDCOVER_CLASSES = {
    10: {"name": "Tree cover", "color": "#34D399"},
    20: {"name": "Shrubland", "color": "#A3E635"},
    30: {"name": "Grassland", "color": "#FACC15"},
    40: {"name": "Cropland", "color": "#F59E0B"},
    50: {"name": "Built-up", "color": "#94A3B8"},
    60: {"name": "Bare / sparse vegetation", "color": "#D6D3D1"},
    80: {"name": "Water", "color": "#22D3EE"},
    90: {"name": "Herbaceous wetland", "color": "#2DD4BF"},
    95: {"name": "Mangroves", "color": "#059669"},
    100: {"name": "Moss and lichen", "color": "#C4B5FD"},
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://verdant-croissant-7eb837.netlify.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
clf = joblib.load(MODEL_PATH)
print("Model loaded.")

catalog = pystac_client.Client.open(CATALOG_URL, modifier=planetary_computer.sign_inplace)


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    bbox = [
        req.lon - BUFFER_DEG,
        req.lat - BUFFER_DEG,
        req.lon + BUFFER_DEG,
        req.lat + BUFFER_DEG,
    ]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise HTTPException(status_code=404, detail="No recent cloud-free imagery found for this location.")
    item = items[0]

    try:
        bands = {}
        for band in ["B02", "B03", "B04", "B08"]:
            da = rioxarray.open_rasterio(item.assets[band].href)
            clipped = da.rio.clip_box(*bbox, crs="EPSG:4326")
            bands[band] = clipped.squeeze()

        blue = bands["B02"].values.astype("float32")
        green = bands["B03"].values.astype("float32")
        red = bands["B04"].values.astype("float32")
        nir = bands["B08"].values.astype("float32")
        ndvi = (nir - red) / (nir + red + 1e-6)

        X = np.stack([blue, green, red, nir, ndvi], axis=-1).reshape(-1, 5)
        valid = ~np.isnan(X).any(axis=1)

        preds = clf.predict(X[valid])
        unique, counts = np.unique(preds, return_counts=True)
        total = counts.sum()

        land_cover = []
        for cls, count in zip(unique, counts):
            if cls in WORLDCOVER_CLASSES:
                info = WORLDCOVER_CLASSES[cls]
                land_cover.append({
                    "name": info["name"],
                    "value": round(float(count) / total * 100, 1),
                    "color": info["color"],
                })
        land_cover.sort(key=lambda c: c["value"], reverse=True)

        avg_ndvi = float(np.nanmean(ndvi))

        return {
            "coordinates": {"lat": req.lat, "lon": req.lon},
            "scene_date": item.properties.get("datetime"),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "land_cover": land_cover,
            "avg_ndvi": round(avg_ndvi, 3),
            "vegetation_health": (
                "healthy" if avg_ndvi > 0.5 else "moderate" if avg_ndvi > 0.2 else "sparse/stressed"
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")