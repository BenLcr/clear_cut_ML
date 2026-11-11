from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from shapely.geometry import shape

# SETTINGS
# The first and the last date of the images 
first_year = 2018
last_year = 2023

# the first image gives the grid (CRS, pixel size, extent) everything is aligned on
ref_file = "outputs/sentinel/S2B_MSIL2A_20170823T103019_R108_T31UGP_20210201T081140_B04.tif"

# Modify this part depending on the study area and period :)
dates = ["2017-08-01/2017-08-31", "2023-08-01/2023-08-31"]   # before / after, each a day or a period like "2017-08-01/2017-08-31"
bbox = (6.65, 47.85, 6.95, 48.10)   # lon_min, lat_min, lon_max, lat_max (EPSG:4326)
bands = ["B02", "B03", "B04", "B08", "B11"]

# DIRECTORIES
workdir = Path("outputs/")
out_dir = workdir / "results"
out_dir.mkdir(exist_ok=True)
out_sentinel = workdir / "sentinel"
catalog_file = workdir / "hansen_contours.gpkg"
dataset_file = workdir / "dataset.npz"
bands_dir = workdir / "sentinel"
outlines_dir = workdir / "outlines"

dst_crs = "EPSG:32632"
res = 10
seed = 200

scenes = {
    2017: "S2B_MSIL2A_20170823T103019_R108_T31UGP_20210201T081140",  # TO MODIFY
    2023: "S2A_MSIL2A_20230810T103631_R008_T32ULU_20241022T015940",
}

offset = {2017: 0, 2023: 1000}   # processing baseline >= 04.00 adds 1000 to the values

# FUNCTIONS 

def make_grid(cat, margin=500):
    # grid with 10 m pixels covering all the polygons of the catalog plus margin meters around
    xmin, ymin, xmax, ymax = cat.total_bounds
    x0 = (xmin - margin) // res * res
    y1 = (ymax + margin) // res * res + res
    W = int((xmax + margin - x0) // res) + 1
    H = int((y1 - (ymin - margin)) // res) + 1
    return H, W, from_origin(x0, y1, res, res)

def load_band(file, offset, H, W, transform):
    # read one band and put it on the common grid
    dn = np.zeros((H, W), "float32")
    with rasterio.open(file) as ds:
        reproject(rasterio.band(ds, 1), dn, dst_transform=transform, dst_crs=dst_crs,
                  src_nodata=0, dst_nodata=0, resampling=Resampling.bilinear)
    # reflectance = (DN - offset) / 10000, nodata becomes NaN
    return np.where(dn > 0, np.clip((dn - offset) / 10000, 0, 1), np.nan)

def scores(label, m, score_map, thr):
    # pixel scores on the pixels selected by m, a pixel is "cut" if its score >= thr
    y_true = label[m]
    pred = score_map[m] >= thr
    tp = (y_true & pred).sum()
    fp = (~y_true & pred).sum()
    fn = (y_true & ~pred).sum()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp / max(tp + fp + fn, 1)

def contours(detected, transform, min_pixels):
    # group connected detected pixels, drop the tiny ones, turn the rest into polygons
    lbl, n = ndi.label(detected, structure=np.ones((3, 3)))
    sizes = np.bincount(lbl.ravel())
    big = sizes >= min_pixels
    big[0] = False                      # 0 is the background
    lbl = np.where(big[lbl], lbl, 0)

    polys = [shape(g) for g, v in shapes(lbl.astype("int32"), mask=lbl > 0, transform=transform, connectivity=8)]
    det = gpd.GeoDataFrame(geometry=polys, crs=dst_crs)
    det["area_ha"] = det.area / 1e4
    return det

def load_dataset(file):
    data = np.load(file)
    X = data["X"].astype("float32", copy=False)
    feats = [str(f) for f in data["feats"]]
    transform = Affine(*data["transform"])
    return X, data["label"], data["mask"], data["sets"], feats, transform

def build_features(H, W, transform):
    pre = {b: load_band(bands_dir / (scenes[2017] + "_" + b + ".tif"), offset[2017], H, W, transform) for b in bands}
    post = {b: load_band(bands_dir / (scenes[2023] + "_" + b + ".tif"), offset[2023], H, W, transform) for b in bands}

    ndvi_pre = (pre["B08"] - pre["B04"]) / (pre["B08"] + pre["B04"])
    ndvi_post = (post["B08"] - post["B04"]) / (post["B08"] + post["B04"])

    F = {"NDVI_pre": ndvi_pre, "NDVI_post": ndvi_post, "dNDVI": ndvi_pre - ndvi_post}
    for b in bands:
        F[b + "_post"] = post[b]
        F["d" + b] = pre[b] - post[b]

    feats = list(F)
    X = np.dstack([F[k] for k in feats]).reshape(-1, len(feats))  # one row per pixel
    return X, feats, ndvi_pre, ndvi_post