import bp_sq25_data_acq
import re
import os
import glob
import geopandas as gpd
import pandas as pd

frame = bp_sq25_data_acq.get_coordinates()

land_use_classes = ['urban', 'parks_open_space', 'undeveloped', 'transportation', 'water']

# Collect mix vectors in a list
mix_vectors = []




path = '/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/Rasters/'
for system, gdf in frame.items():
    system_path = os.path.join(path, system)
    for lu_class in land_use_classes:
        gdf[lu_class] = 0.0
    for idx, gdf_row in gdf.iterrows():
        name = re.sub(r'[^a-zA-Z0-9]', '_', gdf_row['stop_name'])[:30]
        print(f'processing {system} - {name}')
        station_path = os.path.join(system_path, name)
        shp = glob.glob(os.path.join(station_path, '*.shp'))

        shp_df = gpd.read_file(shp[0])

        shp_df['area'] = shp_df.geometry.area

        area_by_class = shp_df.groupby('class')['area'].sum()
        total_area = area_by_class.sum()

        # Normalize to proportions
        mix = {cls: area_by_class.get(cls, 0) / total_area for cls in land_use_classes}
        #vector = list(float(mix[cls]) for cls in land_use_classes)
        
        for cls in land_use_classes:
            gdf.loc[idx, cls] = mix[cls]
        #gdf.loc[idx, 'mix_vector'] = vector
    print(gdf)
    save_path = os.path.join('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data', f"{system}.csv")
    gdf.to_csv(save_path, index=False)


        