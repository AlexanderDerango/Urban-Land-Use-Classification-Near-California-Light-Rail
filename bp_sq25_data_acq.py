import geopandas as gpd
from shapely.geometry import Point

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import requests
import lxml.html as lx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from rapidfuzz import process, fuzz

from pystac_client import Client
from planetary_computer import sign
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds, transform
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from shapely.geometry import mapping, shape, box

import os
import json

import concurrent.futures



def get_coordinates():
    """Retrieving station coordinates
        Final attributes: service  - stop_id - stop_name - lat - long - geometry
        https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip"""

    spatial_ref = 'https://spatialreference.org/ref/?search=california' \

    NAIP_REF = 'UTM NAD83 Zone 12 North - EPSG 26912'
    #Reading in sacRT shapefiles to gpd dataframe
    sacRT_rail = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/sacrt_stations/SacRTStops_Rail_0402.shp')

    #Reading in SFMTA Muni Metro shapefiles to gpd dataframe
    muni_metro = pd.read_csv('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/muni_gtfs-current/stops.txt')

    #Reading in VTA light rail shapefile into gpd dataframe
    vta_rail = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/VTA_stations/LRTStn.shp')

    #Reading in LA metro shapefiles to gpd dataframes
    lam_c = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/lametro_stations/803_Green_Stations_0316/GreenLine0316.shp')
    lam_a = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/lametro_stations/230711_All_A-Line_Stations_Post_RC/230711_All_A-Line_Stations_Post_RC.shp')
    lam_e = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/lametro_stations/230711_All_E-Line_Stations_Post_RC/230711_All_E-Line_Stations_Post_RC.shp')
    lam_k = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/lametro_stations/230711_All_K-Line_Stations/230711_All_K-Line_Stations.shp')

    #Reading in San Diego Trolley shapefiles to gpd dataframes
    mts_trolley = gpd.read_file('/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/SDMTS_transit_stops_datasd/transit_stops_datasd.shp')

    #Data cleaning
    #sacRT
    #Selecting columns
    sacRT_rail = sacRT_rail.iloc[:, [0, 2, 4, 5, 7]]
    #removing ' (__)' from stops that have multiple directions

    sacRT_rail['stop_name'] = sacRT_rail['stop_name'].str.replace(r' \([^)]*\)', '', regex=True)
    #Dropping stops that have duplicate service IE different directions, station platforms are spatially separated
    sacRT_rail = sacRT_rail.drop_duplicates(subset='stop_name', keep='first')
    #Standardizing column names
    sacRT_rail.rename(columns = {
        'stop_lat':'lat',
        'stop_lon':'long'
    }, inplace=True)
    #Inserting service type for future df concat.
    sacRT_rail.insert(0, 'service', 'sacRT')
    sacRT_rail = sacRT_rail.to_crs(epsg = 26910)

    #SFMTA
    # Selecting only muni metro transit stops
    #N-Judah, L-Taraval, M-Ocean view, T-Third Street, K-Ingleside, J-Church
    base = 'https://www.sfmta.com/routes/'
    routeList = ['j-church', 'k-ingleside', 'm-ocean-view', 'l-taraval', 'n-judah', 't-third-street']
    stationList = []
    for route in routeList:
        response = requests.get(base+route)
        html = lx.fromstring(response.text)
        path = '//details[@id="inbound"]//table//tbody/tr'
        rows = html.xpath(path)
        for row in rows:
            stop_id = row.xpath('./td[2]/text()')
            stationList.extend(stop_id)
    stationList = list(set(stationList))
    #Only keeping rows that match stationList
    #Standardizing types
    muni_metro['stop_code'] = muni_metro['stop_code'].astype(str).str.strip()
    mask = muni_metro['stop_code'].isin(stationList)
    muni_metro = muni_metro[mask]
    
    #Selecting columns
    muni_metro = muni_metro.iloc[:, [1, 2, 4, 5]]
    #Standardizing column names
    muni_metro.rename(columns = {
        'stop_lat':'lat',
        'stop_lon':'long',
        'stop_code':'stop_id'
    }, inplace=True)
    # create geom - PCS point for California State Plane
    # Bay Area State plane code - 2227
    geom = gpd.points_from_xy(muni_metro['long'], muni_metro['lat'])
    muni_metro = gpd.GeoDataFrame(muni_metro, geometry=geom, crs='EPSG:4326')

    #Reprojecting into California State Plane System (Bay Area, Zone _)
    muni_metro = muni_metro.to_crs(epsg=26910)
    muni_metro.insert(0, 'service', 'SFMTA')


    #VTA
    vta_rail = vta_rail.iloc[:, [0, 1, 2, 4]]
    vta_rail.rename(columns = {
        'LONG_':'long',
        'LAT':'lat',
        'STA_NAME':'stop_name'
    }, inplace=True)
    vta_rail['stop_id'] = pd.Series(range(1, len(vta_rail)+1))
    vta_rail.insert(0, 'service', 'VTA')
    vta_rail.dropna(inplace=True)
    vta_rail = vta_rail.to_crs(epsg=26910)
    vta_rail=vta_rail.reindex(columns=['service', 'stop_id', 'stop_name', 'long', 'lat', 'geometry', 'coords'])
    


    #LA Metro
    #Selecting and renaming c line columns, which are different
    lam_c = lam_c.iloc[:, [6, 7, 8, 9, 13]]
    lam_c.rename(columns ={
        'STOPNUM':'STOP_ID',
        "STATION":'STOP_NAME',
        'LAT':'STOP_LAT',
        'LONG':'STOP_LON'
    }, inplace=True)
    #Combining light rail line columns
    lam = pd.concat([lam_c, lam_k, lam_e, lam_a])
    lam.rename(columns = {
        'STOP_ID':'stop_id',
        'STOP_NAME':'stop_name',
        'STOP_LAT':'lat',
        'STOP_LON':'long'
    }, inplace=True)
    lam.insert(0, 'service', 'LAM')
    #Reprojecting from WGS84 GCS to California State Plane Zone 5 State Plane
    lam = lam.to_crs(epsg=26911)
    


    #SD MTS - Selecting only San Diego Trolley Stops
    mts_trolley = mts_trolley.iloc[:, [2, 4, 5, 6, 14]]
    #mts_trolley = mts_trolley[~mts_trolley['stop_id'].str.match(r'^\d{5}$', na=False)]

    #Get stop codes
    # url = 'https://www.sdmts.com/getting-around/departures-and-schedules'
    # response = requests.get(url)
    # html = lx.fromstring(response.text)
    # lines = [510, 535, 530, 520]
    # for line in lines:
    #     xpath = f'//div[@class="routes-list"]//li[@tabindex="{line}"]'

    # url = 'https://www.sdmts.com/transit-services/trolley#'
    # response = requests.get(url)
    # html = lx.fromstring(response.text)
    # stationList = []
    # #Build list of light rail stations from tabs on website via xpath
    # for i in range(1, 5):
    #     xpath = f'//div[@x-show="openTab === \'{i}\'"]//table//tbody//tr//td[1]/text()'
    #     stationList.extend(html.xpath(xpath))
    # #clean and standardize station names, remove duplicates
    # stationList  = list(set(station.strip()  for station in stationList if station and station.strip()))
    # for station in stationList:
    #     print(station)
    # print(test)
    stop_ids = ['75000', '75002', '75004', '75006', '75008,' '75010',
                '75012', '75014', '75016', '75106', '75104', '75018', '75102',
                '75092', '75090', '75088', '75087', '75085', '75083', '75080', 
                '75078', '75076', '75042', '77771', '77773', '77775', '77778', '77780',
                '77782', '77784', '77786', '77787', '75020', '75023', '75025',
                '77790', '75098', '75096', '75095', '75083', '75080',
                '75078', '75076', '75042', '75044', '75047', '75048', 
                '75050', '75053', '75055', '75056','75059', '75060', '75063',
                '75064', '75032', '75030', '75029', '75026', '75074', '75073',
                '75070', '75069', '75067', '75040', '75038', '75036', '75034', '75030',
                '75029','75027']
    stop_ids = list(set(stop_ids))
 
    mask = mts_trolley['stop_id'].isin(stop_ids)
    mts_trolley = mts_trolley[mask]


    # def clean_name(name):
    #     name = name.lower()
    #     name = re.sub(r'station$|stop$|stn$', '', name).strip()  # remove common suffixes
    #     name = re.sub(r'street$','st', name)
    #     name = re.sub(r'avenue$','av', name)
    #     name = re.sub(r'road$','rd', name)
    #     name = re.sub(r'[^a-z0-9\s]', '', name)  # remove punctuation
    #     name = re.sub(r'\s+', ' ', name).strip()  # normalize whitespace
    #     return name

    # stationList = [clean_name(station) for station in stationList]
    # mts_trolley['stop_name'] = mts_trolley['stop_name'].apply(clean_name)
    # #Naming conventions differ from GTFS data and scraped station name data on website, as well as station codes
    # #Use fuzzy string matching instead
    # #Define subfunction
    # def station_matches(stop_name, stop_list, threshold=84):
    #     clean_stop = clean_name(stop_name)
    #     if clean_stop in stop_list:
    #         return stop_name 
    #     #use token_sort_ratio for scoring incase station is out of order
    #     match, score, _ = process.extractOne(stop_name, stop_list, scorer=fuzz.token_sort_ratio)
    #     return match if score >= threshold else None
    # #Apply fuzzy matching function to each row in geodataframe - vectorized operation
    # mts_trolley['match'] = mts_trolley['stop_name'].apply(
    #     lambda x: station_matches(x, stationList)
    # )
    # #Filter dataframe based on non-NA match column and drop that column
    # mts_trolley = mts_trolley[mts_trolley['match'].notna()].drop('match', axis=1)
    #drop remaining duplicates from route overlap
    mts_trolley = mts_trolley.drop_duplicates(subset='stop_name', keep='first')
    #fix column name, add service
    mts_trolley.rename(columns={
        'stop_lat':'lat',
        'stop_lon':'long'
    }, inplace=True)
    mts_trolley.insert(0, 'service', 'MTS')
    mts_trolley= mts_trolley.to_crs(epsg=26911)
    
    #Combining all gdfs
    stations = {
        'sacRT':sacRT_rail, 
        'SFMTA':muni_metro, 
        'VTA':vta_rail, 
        'LAM':lam, 
        'MTS':mts_trolley}
    #Confirm data type and reproject CRS
    #Store coordinates and geometry buffer for raster clipping
    # print(isinstance(stations, gpd.GeoDataFrame))
    for system in stations.values():
        system['coords']=system.geometry.to_crs(epsg=4326)
        #400m buffer or roughly 0.25 miles around station
        system['geometry'] = system.geometry.buffer(400)
    
    return stations
    

def get_rasters(stations, output_dir = 'Rasters'):
    def get_naip_tiles(system, client):
        """Retrieves all tiles within system extent"""
        bounds = list(system.total_bounds)
        search = client.search(
            collections=['naip'],
            bbox=bounds,
            max_items=100
        )
        return list(search.items())
    def get_station_raster(gdf_row, system, naip_tiles, base_dir):
        """Retrieve and store individual station raster"""
        #Retrieve/create directory if necessary
        system_dir=os.path.join(base_dir, system)
        os.makedirs(system_dir, exist_ok=True)

        #Get stop name and build output path
        name= re.sub(r'[^a-zA-Z0-9]', '_', gdf_row['stop_name'])[:30]
        output_path=os.path.join(system_dir, f"NAIP_{system}_{name}_400m.tif")
        #Avoids naming conflicts
        if os.path.exists(output_path) and False:
            return output_path
        
        station_geom = gdf_row['geometry']
        buffer_deg = 0.0036
        coords = gdf_row['coords']

        naip_tiles_sorted = sorted(naip_tiles, 
                          key=lambda x: x.properties.get("datetime", "1900-01-01"), 
                          reverse=True)
        newest_year = max(t.properties.get("naip:year", 0) for t in naip_tiles_sorted)
        newest_tiles = [t for t in naip_tiles_sorted if t.properties.get("naip:year", 0) == newest_year]

        tiles = [tile for tile in newest_tiles if shape(tile.geometry).contains(gdf_row['coords'].buffer(0.0036))]
        if not tiles:
            print(f"No single-tile NAIP coverage for {system}/{name}, trying intersecting tiles...")
            tiles = [tile for tile in newest_tiles if shape(tile.geometry).intersects(gdf_row['coords'].buffer(0.0036))]

            if not tiles:
                print(f"No NAIP coverage at all for {system}/{name}")
                return None
            try:
                #tiles.sort(key=lambda x: x.properties.get("naip:year", 0), reverse=True)
                signed_tiles = [sign(tile) for tile in tiles]
                srcs = []
                for t in signed_tiles:
                    try:
                        src = rasterio.open(t.assets['image'].href)
                        srcs.append(src)
                    except Exception as e:
                        print(f"Error opening tile {t.id}: {e}")
                        continue
                
                if not srcs:
                    print(f"No valid tiles opened for {system}-{name}")
                    return None
                
                x, y = coords.x, coords.y
                square_bounds_wgs84 = (x - buffer_deg, y - buffer_deg, x + buffer_deg, y + buffer_deg)
                
                # Convert WGS84 bounds to the CRS of first source
                square_bounds = transform_bounds('EPSG:4326', srcs[0].crs, 
                                            square_bounds_wgs84[0], square_bounds_wgs84[1], 
                                            square_bounds_wgs84[2], square_bounds_wgs84[3])
                intersecting_srcs = []
                for i, src in enumerate(srcs):
                    src_bounds = src.bounds
                    # Check if any part of the source overlaps with square bounds
                    if (src_bounds[0] < square_bounds[2] and src_bounds[2] > square_bounds[0] and
                    src_bounds[1] < square_bounds[3] and src_bounds[3] > square_bounds[1]):
                        intersecting_srcs.append(src)

                    if not intersecting_srcs:
                        print(f"No sources intersect square bounds for {system}-{name}")
                        return None
                    
                print('merging tiles')
                mosaic, out_transform = merge(
                    intersecting_srcs,
                    bounds=station_geom.bounds,
                    res=(intersecting_srcs[0].transform.a, -intersecting_srcs[0].transform.e),  # Exact pixel size
                    method='first',  # Prevents averaging of pixel values
                    dtype=intersecting_srcs[0].dtypes[0]  # Maintain original data type
                )
                mosaic_meta = intersecting_srcs[0].meta.copy()
                mosaic_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "compress": "DEFLATE",
                    "tiled": True,
                    "predictor": 2,
                    'zlevel': 6,
                    "blockxsize": 256,
                    "blockysize": 256,
                })
                
                with rasterio.open(output_path, "w", **mosaic_meta) as dest:
                    dest.write(mosaic)
                print(f'Successfully mosaicked {system}-{name}')
                return output_path
            except Exception as e:
                print(f"Failed mosaicking for {system}-{name}: {str(e)}")
                return None
            finally:
                for src in srcs:
                    src.close()
        
        try:
            tiles.sort(key=lambda x: x.properties.get("naip:year", 0), reverse=True)
            item = sign(tiles[0]) 
            
            asset_href = item.assets["image"].href

            with rasterio.open(asset_href) as src:
                #Use a window to only read station bounds for efficiency
                window = from_bounds(*station_geom.bounds, transform=src.transform)
                out_image = src.read(window=window)

                #Metadata and internal efficient file storage
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": window.height,
                    "width": window.width,
                    "transform": src.window_transform(window),
                    "compress": "DEFLATE",
                    "tiled": True,
                    "predictor": 2,
                    'zlevel':6,
                    "blockxsize": 256, 
                    "blockysize": 256,
                })
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
            print(f'Successfully retrieved {system}-{name}')
            return output_path
        except Exception as e:
            print(f"Failed on {system}-{name}: {str(e)}")
            return None

    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    #Retrieve NAIP tiles for specific system:
    system_tiles = {}
    for system, gdf in stations.items():
        print('NAIP tile fetch for:', system)
        temp_gdf = gdf.to_crs(epsg=4326)
        system_tiles[system]= get_naip_tiles(temp_gdf, client)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for system, gdf in stations.items():
            for _, station_row in gdf.iterrows():
                futures.append(
                    executor.submit(
                        get_station_raster,
                        station_row,
                        system,
                        system_tiles[system],
                        output_dir
                    )
                )
        
        # Wait for all tasks to complete
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    print(f"Processed {len([r for r in results if r])} stations successfully")
    return results

def test_raster(stations, idx=0):
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    geometry = stations.iloc[idx, 6]
    full_geom= stations.iloc[idx, 5]
    search = client.search(
    collections=["naip"],
    intersects=geometry.__geo_interface__,
    #query={"year": {"eq": 2024}},
    max_items=3
    )

    items = list(search.items())
    if not items:
        raise ValueError("No NAIP imagery found for this location.")
    item = sign(items[0]) 
    

    asset_href = item.assets["image"].href

    with rasterio.open(asset_href) as src:
        print(src.crs)
        out_image, out_transform = mask(src, [mapping(full_geom)], crop=True)
        out_meta = src.meta.copy()

    # Update metadata for saving
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
        "tiled": True,
        "predictor": 2
    })
    
    plt.imshow(out_image[:3].transpose(1, 2, 0))  # RGB
    plt.title("NAIP 1km Buffer")
    plt.axis('off')
    plt.show()
    




mosaic = """
Successfully mosaicked MTS-pacific_fleet - 41
Successfully mosaicked sacRT-Sacramento_Valley_Station - 48
Successfully mosaicked LAM-Westchester___Veterans_Station - 16
Successfully mosaicked VTA-Vienna - 21
Failed mosaicking for VTA-Tamien: Read failed. See previous exception for details. - 17
Successfully mosaicked MTS-32nd_commercial_st - 3
Successfully mosaicked MTS-va_medical_center - 55
Successfully mosaicked MTS-executive_drive - 21
Successfully mosaicked LAM-Artesia_Station - 59
"""

# stations = get_coordinates()
# print(stations['MTS'].iloc[1,:])
# print(stations['MTS'].iloc[1,:]['stop_name'])
# stations = {
#     'MTS': stations['MTS']
# }
# for i in stations['MTS'].iloc[:,'stop_name']:
#     print(i)
# substations = {
#     'sacRT': gpd.GeoDataFrame(stations['sacRT'].iloc[[48],:], geometry='geometry', crs=stations['sacRT'].crs),
#     'VTA':gpd.GeoDataFrame(stations['VTA'].iloc[[17, 21],:], geometry='geometry',crs=stations['VTA'].crs),
#     'LAM':gpd.GeoDataFrame(stations['LAM'].iloc[[16, 59],:],geometry='geometry',crs=stations['LAM'].crs),
#     'MTS':gpd.GeoDataFrame(stations['MTS'].iloc[[3, 21, 41, 55],:],geometry='geometry',crs=stations['MTS'].crs)
# }
# test_raster(stations['VTA'], 17)
# for i, t, in enumerate(stations['MTS'].loc[:, 'stop_name']):
    # print(i, t)
# get_rasters(stations)
# get_rasters(substations)

