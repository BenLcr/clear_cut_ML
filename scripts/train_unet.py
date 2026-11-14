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

# STANDARDIZATION
mean = X[train_mask.reshape(-1)].mean(axis=0)
std = X[train_mask.reshape(-1)].std(axis=0)
X_grid = np.nan_to_num((X_grid - mean) / std).astype("float32")
del X, ndvi_post

# PATCH
patch_size = 32
pad = patch_size // 2
X_padded = np.pad(X_grid, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
label_padded = np.pad(label.astype("float32"), ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
del X_grid

# TRAINING PIXELS
pos_idx = np.flatnonzero(train_mask & label)
neg_pool = np.flatnonzero(train_mask & ~label)

N_POS = 5000
N_NEG = 25000
N_VAL = 5000

pos_idx = rng.choice(pos_idx,size=min(len(pos_idx), N_POS),replace=False)
neg_idx = rng.choice(neg_pool,size=min(len(neg_pool), N_NEG),replace=False)

train_idx = np.concatenate([pos_idx, neg_idx])
rng.shuffle(train_idx)

# TRAINING PATCHES
X_train_patches = np.array([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(train_idx // W, train_idx % W)])
y_train_patches = np.array([label_padded[r:r+patch_size, c:c+patch_size] for r, c in zip(train_idx // W, train_idx % W)])
y_train_patches = y_train_patches[..., np.newaxis]

print("X_train:", X_train_patches.shape)
print("y_train:", y_train_patches.shape)

X_train_patches = X_train_patches.astype(np.float32, copy=False)
y_train_patches = y_train_patches.astype(np.float32, copy=False)

# VALIDATION PATCHES
val_pool = np.flatnonzero(val_mask)

val_idx = rng.choice(val_pool,size=min(len(val_pool), N_VAL), replace=False)
X_val_patches = np.array([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(val_idx // W, val_idx % W)])
y_val_patches = np.array([label_padded[r:r+patch_size, c:c+patch_size] for r, c in zip(val_idx // W, val_idx % W)])
y_val_patches = y_val_patches[..., np.newaxis]

print("X_val:", X_val_patches.shape)
print("y_val:", y_val_patches.shape)

X_val_patches = X_val_patches.astype(np.float32, copy=False)
y_val_patches = y_val_patches.astype(np.float32, copy=False)

# UNET
inputs = keras.layers.Input(shape=(patch_size, patch_size, len(feats)))

# Contracting path (Encoder)
conv1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
conv1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
pool1 = keras.layers.MaxPool2D((2,2))(conv1)

conv2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
conv2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
pool2 = keras.layers.MaxPool2D((2,2))(conv2)

conv3 = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
conv3 = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
pool3 = keras.layers.MaxPool2D((2,2))(conv3)

# Bottleneck
b = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
b = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')(b)

# Expanding path
up3 = keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(b)

up3 = keras.layers.Concatenate()([up3, conv3])
up3 = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(up3)
up3 = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(up3)

up2 = keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(up3)

up2 = keras.layers.Concatenate()([up2, conv2])
up2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(up2)
up2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(up2)

up1 = keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(up2)

up1 = keras.layers.Concatenate()([up1, conv1])
up1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(up1)
up1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(up1)

# Output
outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(up1)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])

model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    x=X_train_patches,
    y=y_train_patches,
    verbose=1,
    epochs=30,
    batch_size=32,
    validation_data=(X_val_patches, y_val_patches),
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
batch_size_pred = 1024
preds_list = []

for i in range(0, len(mask_idx), batch_size_pred):
    idx = mask_idx[i : i + batch_size_pred]
    rows, cols = idx // W, idx % W
    batch = np.stack([X_padded[r:r+patch_size, c:c+patch_size, :] for r, c in zip(rows, cols)])
    preds = model.predict(batch, batch_size=1024, verbose=0)[:, pad, pad, 0]
    preds_list.append(preds)

proba[mask] = np.concatenate(preds_list)

proba[mask] = np.concatenate(preds_list)
thresholds = np.arange(0.05, 1.0, 0.05)
ious = [scores(label, val_mask, proba, t)[2] for t in thresholds]
best_threshold = thresholds[int(np.argmax(ious))]

precision, recall, iou = scores(label, test_mask, proba, best_threshold)
print("UNET | threshold", round(best_threshold, 2),
      "| test: precision", round(precision, 3), "recall", round(recall, 3), "IoU", round(iou, 3))

# SAVE MODEL
model.save(str(out_dir / "unet.keras"))
np.savez(out_dir / "unet.npz", mean=mean, std=std, feats=np.array(feats), threshold=best_threshold, patch_size=patch_size)
np.savez(out_dir / "unet.npz", **history.history)
print("model saved in", out_dir / "unet.keras")

# OUTLINE DETECTION
detected = mask & (proba >= best_threshold)
det = contours(detected, transform, MIN_PIXELS)
det.to_file(out_dir / "detections_unet.gpkg", driver="GPKG")
print(len(det), "detections,", round(det["area_ha"].sum()), "ha, saved in", out_dir)