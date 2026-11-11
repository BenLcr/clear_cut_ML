import os
import planetary_computer
import pystac_client
import rioxarray
from functions import out_sentinel, dates, bbox, bands

out_dir = out_sentinel

# connect to the Sentinel-2 L2A catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

scenes = []  # ids of the downloaded scenes, same order as dates

for date in dates:
    # look for scenes covering the bbox on that date
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date)
    items = list(search.items())
    if len(items) == 0:
        print("no scene found for", date)
        continue

    # if there are several, keep the least cloudy one
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    print(item.id)
    scenes.append(item.id)
    print("cloud cover:", item.properties["eo:cloud_cover"], "%")

    # scenes processed with baseline >= 04.00 (after Jan 2022) have +1000 added to the values
    print("processing baseline:", item.properties["s2:processing_baseline"])

    # download each band, cut to the bbox, save as a tif (raw values, not rescaled)
    for band in bands:
        da = rioxarray.open_rasterio(item.assets[band].href)
        da = da.rio.clip_box(*bbox, crs="EPSG:4326")
        da.rio.to_raster(os.path.join(out_dir, item.id + "_" + band + ".tif"))
        print("saved", band)

# the two scenes can be on different tiles (different CRS), so put the second one
# on the grid of the first one to be able to compare them pixel by pixel
if len(scenes) == 2:
    reference = rioxarray.open_rasterio(os.path.join(out_dir, scenes[0] + "_B04.tif"))
    for band in bands:
        da = rioxarray.open_rasterio(os.path.join(out_dir, scenes[1] + "_" + band + ".tif"))
        da = da.rio.reproject_match(reference)
        da.rio.to_raster(os.path.join(out_dir, scenes[1] + "_" + band + "_matched.tif"))
        print("matched", band)

import subprocess

# Create a VRT for each date
for scene in scenes:
    vrt_path = os.path.join(out_dir, scene + ".vrt")

    # Use matched bands when available
    tif_files = []
    for band in bands:
        matched_file = os.path.join(out_dir, scene + "_" + band + "_matched.tif")
        original_file = os.path.join(out_dir, scene + "_" + band + ".tif")

        if os.path.exists(matched_file):
            tif_files.append(matched_file)
        else:
            tif_files.append(original_file)

    subprocess.run(["gdalbuildvrt", "-separate", vrt_path, *tif_files], check=True)

    print("Created:", vrt_path)