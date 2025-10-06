import ee
from pathlib import Path
import pandas as pd
import ast

ee.Authenticate() #Uncomment this whenever needed, once done usually not needed for 1-2 days
ee.Initialize(project='raman-461708')

AEZ_no = 6
# Import the two AEZ_7_boundaries assets
aez_boundaries_0 = ee.FeatureCollection('projects/raman-461708/assets/AEZ_'+str(AEZ_no)+'_boundaries_0')
aez_boundaries_1 = ee.FeatureCollection('projects/raman-461708/assets/AEZ_'+str(AEZ_no)+'_boundaries_1')

# Merge the two feature collections into one
merged_aez_boundaries = aez_boundaries_0.merge(aez_boundaries_1)

import pandas as pd

# Read the CSV file
df = pd.read_csv("AEZ_"+str(AEZ_no)+"/status.csv")

# Function to create a tile (rectangle polygon) from diagonal points
def create_tile_from_points(points):
    # points is a string representation of a tuple, e.g. "((lat1, lon1), (lat2, lon2))"
    if isinstance(points, str):
        points_tuple = ast.literal_eval(points)
    else:
        points_tuple = points
    (lon1, lat1), (lon2, lat2) = points_tuple
    # Get min/max for lat/lon to define the rectangle
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    # Return the four corners in order (clockwise or counterclockwise)
    return [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
        (min_lat, min_lon)  # close the polygon
    ]

# Create a new column 'tile' with the rectangle polygon for each row
df['tile'] = df['points'].apply(create_tile_from_points)
tiles = df['tile'].tolist()

samples = []
import concurrent.futures
import time

# Function to chunk a list into sublists of size p
#def chunk_list(lst, p):
#    return [lst[i:i + p] for i in range(0, len(lst), p)]

# Define the number of tiles per group
#p = 1

# Chunk the tiles into groups
#tile_groups = chunk_list(tiles, p)

def fc_to_df(fc):
    # Get the data from the FeatureCollection
    features = fc.getInfo()['features']
    
    # Extract properties and geometry
    data = [feature['properties'] for feature in features]
    
    return pd.DataFrame(data)

def process_tile(tile, index, one_at_a_time=True):
    tile = ee.FeatureCollection([ee.Geometry.Polygon(tile)])
    
    #aez_region = ee.FeatureCollection("users/mtpictd/agro_eco_regions").filter(ee.Filter.eq("ae_regcode", 7)).filter(tile)
    #g_embs = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate('2024-01-01', '2025-01-01').filterBounds(tile).mosaic()

    mapping = {
        "farm": 1,
        "plantation": 2,
        "scrubland": 3,
        "rest": 0
    }
    reversed_mapping = {v: k for k, v in mapping.items()}
    reversed_ee_mapping = ee.Dictionary(reversed_mapping)

    easy_farm = [
        ee.Filter.gte("rect", 0.67),
        ee.Filter.gt("size", 500),
        ee.Filter.lt("size", 2000),
        ee.Filter.lt("ent", 1)
        ]
    easy_scrubland = [
        ee.Filter.gte("size", 60000)
        ]
    easy_plantation = [
        ee.Filter.lt("area", 20000),
        ee.Filter.gt("area", 1000)
    ]

    all = merged_aez_boundaries.filterBounds(tile)
    farm = all.filter(ee.Filter.And(*easy_farm))
    scrubland = all.filter(ee.Filter.And(easy_scrubland))
    plantation = all.filter(ee.Filter.eq("class", "plantation")).map(lambda x: x.set("area", x.geometry().area())).filter(ee.Filter.And(easy_plantation))
    mapping = {
        "farm": 1,
        "plantation": 2,
        "scrubland": 3,
        "rest": 0
    }
    reversed_mapping = {v: k for k, v in mapping.items()}
    reversed_ee_mapping = ee.Dictionary(reversed_mapping)

    easy_farm = [
        ee.Filter.gte("rect", 0.67),
        ee.Filter.gt("size", 500),
        ee.Filter.lt("size", 2000),
        ee.Filter.lt("ent", 1)
        ]

    easy_scrubland = [
        ee.Filter.gte("size", 60000)
        ]

    
    
    #Filter out farms which doesnot have 3 nearby farms (Removing solo farms inside scrublands)
    farm_buffer = farm.map(lambda x: x.buffer(10))
    # Merge touching farms into clusters
    farm_buffer_union = farm_buffer.union(maxError=1)
    geoms = farm_buffer_union.geometry().dissolve().geometries()
    geoms = geoms.map(lambda geom: ee.Feature(ee.Geometry(geom)))
    # Flatten clusters into individual polygons
    clusters = ee.FeatureCollection(geoms)
    clusters = clusters.map(lambda x: x.set("count", farm.filterBounds(x.geometry()).size())).filter(ee.Filter.lt('count', 3))
    farm = farm.filterBounds(clusters)
    
    #Filtering out scrublands
    lulc_v3 = ee.Image("projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2023_2024").clip(tile.geometry())
    classes_of_interest = [8, 9, 10, 11]
    
    
    # Build a binary mask for classes of interest (1 = of interest, 0 = otherwise)
    interest_mask = (
        lulc_v3
        .remap(classes_of_interest, ee.List.repeat(1, len(classes_of_interest)), 0)
        .rename('interestMask')
        .unmask(0)
    )

    # Pixel area bands
    px_area       = ee.Image.pixelArea().rename('total_area')
    interest_area = px_area.updateMask(interest_mask).rename('interest_area')

    # Stack and clip to the tile
    area_stack = interest_area.addBands(px_area).clip(tile.geometry())

    # Sum both bands over all polygons in one call
    summed = area_stack.reduceRegions(
    collection=scrubland,
    reducer=ee.Reducer.sum(),
    scale=10
    )

    # Compute percentage per feature (no if-condition needed)
    with_percent = summed.map(
        lambda f: f.set(
            'percent_interest',
            ee.Number(f.get('interest_area'))
            .divide(ee.Number(f.get('total_area')))
            .multiply(100)
        )
    )

    # Filter: < 50% interest AND strictly inside the ROI
    scrubland = (
        with_percent
        .filter(ee.Filter.lt('percent_interest', 50))
    )
    # Desired samples per class
    class_values = [(1, 150), (2, 150), (3, 150)]
    class_band = 'label'

    # Build the label raster (priority = last paint wins for overlaps).
    # If you need explicit precedence, keep this order intentional.
    label_image = (
        ee.Image(0).byte().rename(class_band)
        .paint(farm,       mapping["farm"])
        .paint(scrubland,  mapping["scrubland"])
        .paint(plantation, mapping["plantation"])
        .clip(tile.geometry())
    )

    # Prepare lists for stratified sampling
    class_values_list  = [v for v, _ in class_values]
    class_points_list  = [n for _, n in class_values]

    # One stratified sample call; returns up to the requested points per class
    all_samples = label_image.stratifiedSample(
        numPoints=sum(class_points_list),
        classBand=class_band,
        classValues=class_values_list,
        classPoints=class_points_list,
        scale=10,                 # your LULC pixel size
        region=tile.geometry(),
        seed=42,
        dropNulls=True,
        geometries=True
    )
    
    """
    label_image = ee.Image(0).rename("label")
    farm_mask = label_image.clip(farm).mask()
    scrubland_mask = label_image.clip(scrubland).mask()
    plantation_mask = label_image.clip(plantation).mask()

    label_image = label_image.where(farm_mask, mapping["farm"]).where(scrubland_mask, mapping["scrubland"]).where(plantation_mask, mapping["plantation"])
    
    # Classes to sample (exclude background = 0)
    class_values = [(1, 150), (2, 150), (3, 150)]

    # Create masks for all classes and combine them into a single mask
    class_band = 'label'
    class_values_list = [class_val for class_val, _ in class_values]
    class_points_list = [points for _, points in class_values]

    # Sample uniformly from the masked image using a single call to stratifiedSample
    all_samples = label_image.stratifiedSample(
        numPoints=sum(class_points_list),  # Total number of points needed
        classBand=class_band,
        classValues=class_values_list,
        classPoints=class_points_list,
        scale=10,  # Adjust the scale as needed
        region=tile.geometry(),
        seed=42,
        dropNulls=True,
        geometries=True
    )

    # Filter samples to ensure each class has the correct number of points
    def filter_samples(samples, class_val, points):
        return samples.filter(ee.Filter.eq(class_band, class_val)).limit(points)

    filtered_samples_list = [
        filter_samples(all_samples, class_val, points)
        for class_val, points in class_values
    ]

    # Merge all filtered samples into a single FeatureCollection
    all_filtered_samples = ee.FeatureCollection(filtered_samples_list).flatten()
    return all_filtered_samples
    """
    task = ee.batch.Export.table.toDrive(
        collection=all_samples,
        description=str(AEZ_no) + "_tile_" + str(index),
        fileFormat='CSV',
        folder='Scrubland_Field_Delineation/AEZ_' + str(AEZ_no),
        fileNamePrefix='tile_' + str(index),
    )
    task.start()
    print(f"tile_" + str(index) + " started")
    if one_at_a_time:
        while True:
            status = task.status()['state']
            print(f"Task tile_{index} status: {status}")
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(10) 
    #all_filtered_samples_df = fc_to_df(all_filtered_samples)
    #all_filtered_samples_df.to_csv('output_'+str(index)+'.csv', index=False)
    #print("Done tile no ", index)
    

for i in range(15, len(tiles)):
    process_tile(tiles[i], i, one_at_a_time=False)
    
#for i in range(0, len(tiles)):
#    process_tile(tiles[i], i)