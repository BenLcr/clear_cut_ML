import geopandas as gpd
import rioxarray
from rasterio.features import shapes
from shapely.geometry import shape

from functions import first_year, last_year, ref_file

# Hansen tile (10x10 degrees, named by its top-left corner): 50N_000E = 40-50N, 0-10E
tile = "50N_000E"
url = ("https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/"
       "Hansen_GFC-2024-v1.12_lossyear_" + tile + ".tif")

ref = rioxarray.open_rasterio(ref_file).squeeze()
bbox = ref.rio.transform_bounds("EPSG:4326")

loss = rioxarray.open_rasterio(url).squeeze()
loss = loss.rio.clip_box(*bbox)

loss = loss.rio.reproject_match(ref)

years = loss.values
keep = (years >= first_year - 2000) & (years <= last_year - 2000)

mask = loss.copy(data=keep.astype("uint8"))
mask.rio.to_raster("outputs/hansen_mask.tif")

# one polygon per group of connected pixels with the same loss year
rows = []
for geom, value in shapes(years, mask=keep, transform=ref.rio.transform()):
    rows.append({"loss_year": 2000 + int(value), "geometry": shape(geom)})

zones = gpd.GeoDataFrame(rows, crs=ref.rio.crs)
zones["area_ha"] = zones.area / 10000
zones.to_file("outputs/hansen_contours.gpkg", driver="GPKG")

print(len(zones), "polygons,", round(zones["area_ha"].sum(), 1), "ha")