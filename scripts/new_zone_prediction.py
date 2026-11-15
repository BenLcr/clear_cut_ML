import joblib
import argparse
import numpy as np
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from tensorflow import keras

import functions
from functions import bands, build_features, contours, dst_crs, out_dir, res, FOREST_NDVI_MIN, MIN_PIXELS, batch_size_pred

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cnn", choices=["rf", "cnn", "unet"], help="model to use (default: cnn)")
parser.add_argument("--bands-dir", type=Path, default=Path("outputs/new_sentinel"), help="folder of the .tif bands of the new zone")
parser.add_argument("--scene-before", default="S2B_MSIL2A_20170823T103019_R108_T31UGP_20210201T081140", help="name of the scene of the first date, without the band suffix")
parser.add_argument("--scene-after", default="S2A_MSIL2A_20230810T103631_R008_T32ULU_20241022T015940", help="name of the scene of the second date, without the band suffix")
args = parser.parse_args()

# Settings
model_name = args.model
model_dir = Path("outputs/results")

new_bands_dir = args.bands_dir
new_scenes = {2017: args.scene_before, 2023: args.scene_after}
new_offset = {2017: 0, 2023: 1000}

functions.scenes, functions.offset, functions.bands_dir = new_scenes, new_offset, new_bands_dir

#BBOX
extents = []
for year, scene in new_scenes.items():
    file = new_bands_dir / (scene + "_B04.tif")
    with rasterio.open(file) as ds:
        print(year, "|", file.name, "| crs", ds.crs)
        extents.append(transform_bounds(ds.crs, dst_crs, *ds.bounds))
xmin, ymin = max(e[0] for e in extents), max(e[1] for e in extents)
xmax, ymax = min(e[2] for e in extents), min(e[3] for e in extents)

x0 = xmin // res * res
y1 = ymax // res * res
W = int((xmax - x0) // res) + 1
H = int((y1 - (ymin)) // res) + 1
transform = from_origin(x0, y1, res, res)
n_feat = 3 + 2 * len(bands)
print("grid:", H, "x", W, "px")

# FEATURES
X, feats, ndvi_pre, ndvi_post = build_features(H, W, transform)
print(len(feats), "features:", feats)

mask = (ndvi_pre > FOREST_NDVI_MIN) & np.isfinite(X).all(axis=1).reshape(H, W)
valid = np.isfinite(X).all(axis=1).reshape(H, W)
print("valid pixels (both dates):", round(100 * valid.mean(), 1), "% of the grid")
print("forest pixels:", int(mask.sum()), "=", round(100 * mask.mean(), 1), "% of the grid =", round(100 * mask.sum() / max(valid.sum(), 1), 1), "% of the valid pixels")

# PREDICTION
proba = np.full((H, W), np.nan, dtype="float32")

if model_name == "rf":
    saved = joblib.load(model_dir / "random_forest.joblib")
    threshold = saved["threshold"]
    proba[mask] = saved["model"].predict_proba(X[mask.reshape(-1)])[:, 1]

elif model_name == "unet":
    model = keras.models.load_model(model_dir / (model_name + ".keras"))
    flexible = keras.models.clone_model(model, input_tensors=keras.Input(shape=(None, None, model.input_shape[-1])))
    flexible.set_weights(model.get_weights())
    model = flexible
    norm = np.load(model_dir / (model_name + ".npz"))
    threshold = float(norm["threshold"])
    X_grid = np.nan_to_num((X.reshape(H, W, len(feats)) - norm["mean"]) / norm["std"]).astype("float32")
    
    def predict(x):
        h, w = x.shape[:2]
        x = np.pad(x, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)))
        return model.predict(x[None], verbose=0)[0, :h, :w, 0]

    tile, context = 256, 32
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            r0, r1 = max(r - context, 0), min(r + tile + context, H)
            c0, c1 = max(c - context, 0), min(c + tile + context, W)
            p = predict(X_grid[r0:r1, c0:c1])
            proba[r:r + tile, c:c + tile] = p[r - r0:r - r0 + tile, c - c0:c - c0 + tile]
    proba[~mask] = np.nan

elif model_name =="cnn":
    model = keras.models.load_model(model_dir / (model_name + ".keras"))
    norm = np.load(model_dir / (model_name + ".npz"))
    threshold = float(norm["threshold"])
    X_grid = np.nan_to_num((X.reshape(H, W, len(feats)) - norm["mean"]) / norm["std"]).astype("float32")

    patch_size = int(norm["patch_size"])
    pad = patch_size // 2
    X_padded = np.pad(X_grid, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    def get_patches(idx):
        rows, cols = np.unravel_index(idx, (H, W))
        return np.array([X_padded[r:r + patch_size, c:c + patch_size] for r, c in zip(rows, cols)])

    mask_idx = np.flatnonzero(mask)
    preds = [model.predict(get_patches(mask_idx[i:i + batch_size_pred]), verbose=0).ravel() for i in range(0, len(mask_idx), batch_size_pred)]
    proba[mask] = np.concatenate(preds)

else:
    print(model_name, " is unknown.")

# OUTPUTS
detected = np.nan_to_num(proba, nan=0) >= threshold
print(model_name, "| threshold", round(threshold, 2), "| detected pixels:", int(detected.sum()),
      "=", round(detected.sum() * res**2 / 1e4), "ha")

run_name = "newzone_" + model_name
n_run = 1
while (out_dir / (run_name + "_probability.tif")).exists() or (out_dir / (run_name + "_detections.gpkg")).exists():
    n_run += 1
    run_name = f"newzone_{model_name}_{n_run}"
 
profile = dict(driver="GTiff", height=H, width=W, count=1, crs=dst_crs, transform=transform)
with rasterio.open(out_dir / (run_name + "_probability.tif"), "w", dtype="float32", nodata=np.nan, **profile) as dst:
    dst.write(proba, 1)
 
det = contours(detected, transform, MIN_PIXELS)
det.to_file(out_dir / (run_name + "_detections.gpkg"), driver="GPKG")
print(len(det), "detections,", round(det["area_ha"].sum()), "ha, saved in", out_dir, "as", run_name)