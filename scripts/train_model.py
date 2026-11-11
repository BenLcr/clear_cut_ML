import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier

from functions import dataset_file, build_features, catalog_file, outlines_dir, contours, dst_crs, make_grid, out_dir, res, scores, seed

# PARAMETERS
FOREST_NDVI_MIN = 0.6 # a pixel is forest if its NDVI in 2017 is above this
NEG_RATIO = 5         # number of not-cut pixels per cut pixel in the training set
MIN_PIXELS = 25       # smallest detection kept when drawing the contours
rng = np.random.default_rng(seed)

cat = gpd.read_file(catalog_file).to_crs(dst_crs)
H, W, transform = make_grid(cat)

# FEATURES
X, feats, ndvi_pre, ndvi_post = build_features(H, W, transform)
print(len(feats), "features:", feats)

# forest mask
mask = (ndvi_pre > FOREST_NDVI_MIN) & np.isfinite(X).all(axis=1).reshape(H, W)

# SETS: the block grid as a map of pixels, 0 = unused, 1 = train, 2 = val, 3 = test
grid = gpd.read_file(outlines_dir / "spatial_split.gpkg")
set_code = {"train": 1, "validation": 2, "test": 3}
sets = rasterize(((g, set_code[s]) for g, s in zip(grid.geometry, grid["set"]) if s in set_code),
                 out_shape=(H, W), transform=transform, fill=0, dtype="uint8")

# LABELS: label = True where there is a cut.
label = np.zeros((H, W), bool)
for k, name in {1: "train", 2: "val", 3: "test"}.items():
    cuts = gpd.read_file(outlines_dir / f"contours_{name}.gpkg")
    if len(cuts) == 0:
        continue
    cut_px = rasterize(cuts.geometry, out_shape=(H, W), transform=transform, dtype="uint8") > 0
    label |= cut_px
    sets[cut_px] = k

# STATS
print("grid:", H, "x", W, "px | cut pixels:", int(label.sum()), "=", round(label.sum() * res**2 / 1e4), "ha")
print("cut pixels inside the mask:", round(100 * (label & mask).sum() / label.sum(), 1), "%")
for k, name in {1: "train", 2: "val", 3: "test"}.items():
    print("  ", name, ":", int((mask & label & (sets == k)).sum()), "cut pixels /", int((mask & (sets == k)).sum()), "forest pixels")

val = mask & (sets == 2)
test = mask & (sets == 3)

# TRAINING
pos = np.flatnonzero(mask & label & (sets == 1))
neg_pool = np.flatnonzero(mask & ~label & (sets == 1))
neg = rng.choice(neg_pool, size=min(len(neg_pool), NEG_RATIO * len(pos)), replace=False)

idx = np.concatenate([pos, neg])
y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])   # 1 = cut
print("training:", len(pos), "cut /", len(neg), "not cut")

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=seed)
rf.fit(X[idx], y)

for name, importance in sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1]):
    print("  ", name, round(importance, 3))

# probability of clear-cut for every pixel of the mask
proba = np.full((H, W), np.nan, dtype="float32")
proba[mask] = rf.predict_proba(X[mask.reshape(-1)])[:, 1]

# EVALUATION
# Compare the NDVI-drop baseline and the Random Forest using the same evaluation procedure:
# 1) select the detection threshold that maximizes IoU on the validation blocks;
# 2) evaluate the selected threshold once on the independent test blocks.
methods = {
    "NDVI drop": (ndvi_pre-ndvi_post, np.arange(0.05, 1.0, 0.05)),
    "Random Forest": (proba, np.arange(0.05, 1.0, 0.05)),
}
best = {}
for name, (score_map, thresholds) in methods.items():
    ious = [scores(label, val, score_map, t)[2] for t in thresholds]
    best[name] = thresholds[int(np.argmax(ious))]
    precision, recall, iou = scores(label, test, score_map, best[name])
    print(name, "| threshold", round(best[name], 2), "| test: precision", round(precision, 3), "recall", round(recall, 3), "IoU", round(iou, 3))

# SAVE MODEL
joblib.dump({"model": rf, "features": feats}, out_dir / "random_forest.joblib")
print("model saved in", out_dir / "random_forest.joblib")

# OUTPUTS
detected = np.nan_to_num(proba, nan=0) >= best["Random Forest"]

profile = dict(driver="GTiff", height=H, width=W, count=1, crs=dst_crs, transform=transform)
with rasterio.open(out_dir / "cut_probability.tif", "w", dtype="float32", nodata=np.nan, **profile) as dst:
    dst.write(proba, 1)
with rasterio.open(out_dir / "cut_detection.tif", "w", dtype="uint8", **profile) as dst:
    dst.write(detected.astype("uint8"), 1)

train_sample = np.zeros(H * W, "uint8")   # pixels given to the model: 1 = cut, 2 = not cut
train_sample[pos] = 1
train_sample[neg] = 2

dataset_bands = {"forest_mask": mask, "label": label, "sets": sets, "train_sample": train_sample.reshape(H, W)}
with rasterio.open(out_dir / "dataset_masks.tif", "w", dtype="uint8", **{**profile, "count": len(dataset_bands)}) as dst:
    dst.write(np.stack(list(dataset_bands.values())).astype("uint8"))
    dst.descriptions = tuple(dataset_bands)

# the same dataset as arrays, to reload it in a notebook without recomputing anything
np.savez(dataset_file, X=X, mask=mask, label=label, sets=sets, train_sample=train_sample.reshape(H, W),
         feats=np.array(feats), transform=np.array(tuple(transform)[:6]))
print("dataset saved in", dataset_file)

# OUTLINE DETECTION
det = contours(detected, transform, MIN_PIXELS)
det.to_file(out_dir / "detections_rf.gpkg", driver="GPKG")
print(len(det), "detections,", round(det["area_ha"].sum()), "ha, saved in", out_dir)