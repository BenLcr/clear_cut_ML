import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import box

from functions import dst_crs, make_grid, seed, workdir, catalog_file, dataset_file, outlines_dir

# Settings
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15

block_size = 1000       # m, size of the square val / test blocks
rng = np.random.default_rng(seed)

contours_ref_gdf = gpd.read_file(catalog_file).to_crs(dst_crs)
H, W, transform = make_grid(contours_ref_gdf)

# Extent of the outlines
xmin, ymin, xmax, ymax = contours_ref_gdf.total_bounds
print("Bounds :", xmin, ymin, xmax, ymax)

# Grid creation
xs = np.arange(xmin, xmax + block_size, block_size)
ys = np.arange(ymin, ymax + block_size, block_size)

grid = []

for x in xs[:-1]:
    for y in ys[:-1]:
        grid.append(box(x, y, x + block_size, y + block_size))

grid = gpd.GeoDataFrame({"geometry": grid}, crs=contours_ref_gdf.crs)

# Atrribution of each cut to a block
contours_ref_gdf["centroid"] = contours_ref_gdf.geometry.centroid
joined = gpd.sjoin(contours_ref_gdf.set_geometry("centroid"), grid, predicate="within")
contours_ref_gdf["block"] = joined["index_right"]
contours_ref_gdf = contours_ref_gdf.drop(columns="centroid")
blocks = grid.index.to_numpy().copy()
rng.shuffle(blocks)

n = len(blocks)

n_train = int(TRAIN_SPLIT * n)
n_val = int(VALIDATION_SPLIT * n)

train_blocks = blocks[:n_train]
val_blocks = blocks[n_train:n_train + n_val]
test_blocks = blocks[n_train + n_val:]

# Training split
train = contours_ref_gdf[contours_ref_gdf["block"].isin(train_blocks)].copy()
val = contours_ref_gdf[contours_ref_gdf["block"].isin(val_blocks)].copy()
test = contours_ref_gdf[contours_ref_gdf["block"].isin(test_blocks)].copy()

print("Train :", len(train))
print("Validation :", len(val))
print("Test :", len(test))

train.to_file(outlines_dir / "contours_train.gpkg", driver="GPKG")
print("Train dataset saved in", outlines_dir / "contours_train.gpkg")
val.to_file(outlines_dir / "contours_val.gpkg", driver="GPKG")
print("Validation dataset saved in", outlines_dir / "contours_val.gpkg")
test.to_file(outlines_dir / "contours_test.gpkg", driver="GPKG")
print("Test dataset saved in", outlines_dir / "contours_test.gpkg")