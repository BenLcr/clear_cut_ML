import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize
import tensorflow as tf
from tensorflow import keras

from functions import dataset_file, build_features, catalog_file, outlines_dir, contours, dst_crs, make_grid, out_dir, res, scores, seed

NEG_RATIO = 5
FOREST_NDVI_MIN = 0.6
MIN_PIXELS = 5

rng = np.random.default_rng(seed)

cat = gpd.read_file(catalog_file).to_crs(dst_crs)
H, W, transform = make_grid(cat)

# FEATURES
X, feats, ndvi_pre, ndvi_post = build_features(H, W, transform)
print(len(feats), "features:", feats)

X_grid = X.reshape(H, W, len(feats))

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

train_mask = mask & (sets == 1)
val_mask = mask & (sets == 2)
test_mask = mask & (sets == 3)

# TRAINING PIXELS
pos_idx = np.flatnonzero(train_mask & label)
neg_pool = np.flatnonzero(train_mask & ~label)
neg_idx = rng.choice(neg_pool, size=min(len(neg_pool), NEG_RATIO * len(pos_idx)), replace=False)
train_idx = np.concatenate([pos_idx, neg_idx])
rng.shuffle(train_idx)

# STANDARDIZATION
mean = X[train_mask.reshape(-1)].mean(axis=0)
std = X[train_mask.reshape(-1)].std(axis=0)
X_grid = np.nan_to_num((X_grid - mean) / std).astype("float32")

# PATCH
patch_size = 7
pad = patch_size // 2
X_padded = np.pad(X_grid, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

X_train_patches = np.array([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(train_idx // W, train_idx % W)])
y_train = label.ravel()[train_idx].astype("float32")

val_idx = np.flatnonzero(val_mask)
X_val_patches = np.array([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(val_idx // W, val_idx % W)])
y_val = label.ravel()[val_idx].astype("float32")

# CNN
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(patch_size, patch_size, len(feats))))
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) # 0 = no cut, 1 = cut

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])

model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    x=X_train_patches,
    y=y_train,
    verbose=1,
    epochs=30,
    batch_size=256,
    validation_data=(X_val_patches, y_val),
    callbacks=[early_stop]
)

# HISTORY
loss = history.history["loss"]
val_loss = history.history["val_loss"]

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

epochs = range(1, len(loss) + 1)

# EVALUATION
mask_idx = np.flatnonzero(mask)
proba = np.full((H, W), np.nan, dtype="float32")

# BATCH PREDICTION
batch_size_pred = 65536
preds_list = []

for i in range(0, len(mask_idx), batch_size_pred):
    batch_indices = mask_idx[i : i + batch_size_pred]
    batch_patches = np.array([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(batch_indices // W, batch_indices % W)])
    preds_batch = model.predict(batch_patches, verbose=0).ravel()
    preds_list.append(preds_batch)

proba[mask] = np.concatenate(preds_list)
thresholds = np.arange(0.05, 1.0, 0.05)
ious = [scores(label, val_mask, proba, t)[2] for t in thresholds]
best_threshold = thresholds[int(np.argmax(ious))]

precision, recall, iou = scores(label, test_mask, proba, best_threshold)
print("CNN | threshold", round(best_threshold, 2),
      "| test: precision", round(precision, 3), "recall", round(recall, 3), "IoU", round(iou, 3))

# SAVE MODEL
model.save(str(out_dir / "cnn.keras"))
np.savez(out_dir / "cnn.npz", mean=mean, std=std, feats=np.array(feats), threshold=best_threshold, patch_size=patch_size)
np.savez(out_dir / "cnn_history.npz", **history.history)
print("model saved in", out_dir / "cnn.keras")

# OUTLINE DETECTION
detected = mask & (proba >= best_threshold)
det = contours(detected, transform, MIN_PIXELS)
det.to_file(out_dir / "detections_cnn.gpkg", driver="GPKG")
print(len(det), "detections,", round(det["area_ha"].sum()), "ha, saved in", out_dir)