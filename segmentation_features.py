import geopandas as gpd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from rapidfuzz import process, fuzz

from pystac_client import Client

import rasterio

from skimage.restoration import inpaint
from skimage.filters import gaussian
from skimage.segmentation import felzenszwalb, find_boundaries

from skimage.exposure import equalize_hist
from skimage.restoration import inpaint
from skimage.filters import gaussian
from skimage.segmentation import felzenszwalb, find_boundaries

import os
import shutil
import json

import concurrent.futures

from skimage.measure import regionprops
from collections import Counter


import bp_sq25_data_acq


def preprocess_raster(gdf_row, system, path):
        name = re.sub(r'[^a-zA-Z0-9]', '_', gdf_row['stop_name'])[:30]
        raster = f"NAIP_{system}_{name}_400m.tif"
        img_path = path + system + "/" + raster

        with rasterio.open(img_path) as src:
            img = src.read()
            image = src.read().astype("float32")
            transform = src.transform  
            crs=src.crs
            
            # Normalize bands
            for band in range(image.shape[0]):  
                image[band] = (image[band] - np.min(image[band])) / (np.max(image[band]) - np.min(image[band]) + 1e-10)
            
            def white_balance(img):
                img_balanced = img.copy()
                for band in range(3):  # Only for RGB (not NIR)
                    img_balanced[band] = equalize_hist(img[band])
                return img_balanced

            img_clean = white_balance(image) 

            return img_clean, transform, crs, img_path

def object_segmentation(img, fs_scale, fs_sigma, fs_min_size):
    if img.ndim == 3 and img.shape[0] <= 5 and img.shape[0] <= img.shape[1] and img.shape[0] <= img.shape[2]:
        # Likely (bands, height, width) format
        processed_img = np.transpose(img, (1, 2, 0))
        processed_img = gaussian(processed_img, sigma=3.5, channel_axis=-1)
    else:
        processed_img = img.copy()
    
    segments = felzenszwalb(processed_img, scale = fs_scale, sigma=fs_sigma, min_size=fs_min_size)
    return segments

def merge_small_segments(segments, min_size=200):
    H, W = segments.shape
    merged = segments.copy()
    props  = regionprops(segments)
    
    # Precompute neighbor lookups
    # pad to easily sample 4‐neighbors of border pixels:
    padded = np.pad(segments, 1, mode='constant', constant_values=0)
    
    for prop in props:
        lbl, area = prop.label, prop.area
        if area >= min_size:
            continue
        
        # get the coordinates of this small region
        coords = prop.coords + 1  # account for pad
        neighs = []
        for (r, c) in coords:
            # sample 4-connectivity neighbors in padded array
            neighs.extend([
                padded[r-1, c],
                padded[r+1, c],
                padded[r, c-1],
                padded[r, c+1]
            ])
        # exclude self and background
        neighs = [n for n in neighs if n not in (0, lbl)]
        if not neighs:
            continue
        
        # pick the most common neighbor label
        target = Counter(neighs).most_common(1)[0][0]
        merged[segments == lbl] = target
    
    # finally relabel to keep labels consecutive
    unique = np.unique(merged)
    remap  = {old: new for new, old in enumerate(unique)}
    out    = np.vectorize(remap.get)(merged)
    return out

from skimage.draw import polygon
from scipy import ndimage
from collections import Counter
from skimage.measure import regionprops, find_contours, approximate_polygon
import cv2

def fill_unassigned_pixels(smoothed, original, unassigned):
    """Fill unassigned pixels with the most common neighboring label."""
    filled = smoothed.copy()
    
    # Label unassigned regions
    labeled_unassigned, num_regions = ndimage.label(unassigned)
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_unassigned == region_id)
        
        # Dilate to find neighbors
        dilated = ndimage.binary_dilation(region_mask)
        neighbor_mask = dilated & ~region_mask
        
        # Find most common neighbor label (excluding 0)
        neighbor_labels = smoothed[neighbor_mask]
        neighbor_labels = neighbor_labels[neighbor_labels > 0]
        
        if len(neighbor_labels) > 0:
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            filled[region_mask] = most_common
        else:
            # If no neighbors found, use original segmentation
            filled[region_mask] = original[region_mask]
    
    return filled

def smooth_segment_boundaries(segments, tolerance=2.0, morph_size=3):
    """
    Smooth segment boundaries using polygon simplification and morphological operations.
    
    Parameters:
    -----------
    segments : ndarray
        Input segmentation label map
    tolerance : float
        Tolerance parameter for polygon simplification (higher = more simplification)
    morph_size : int
        Size of morphological kernel for smoothing operations
    
    Returns:
    --------
    smoothed_segments : ndarray
        Simplified and smoothed segmentation with same dimensions as input
    """
    H, W = segments.shape
    smoothed = np.zeros_like(segments)
    
    # Get unique labels, excluding background if it's 0
    labels = np.unique(segments)
    if labels[0] == 0:
        labels = labels[1:]  # Skip background
    
    for label in labels:
        # Create binary mask for this segment
        mask = (segments == label)
        
        # Apply morphological operations to smooth boundaries
        kernel = np.ones((morph_size, morph_size), np.uint8)
        # Opening removes small protrusions
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        # Closing fills small intrusions
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of the mask
        contours = find_contours(closed, 0.5)
        
        if not contours:
            # If no contours found, just use the original mask
            smoothed[mask] = label
            continue
        
        # Use the largest contour
        contour = max(contours, key=len)
        
        # Simplify the polygon with Douglas-Peucker algorithm
        simplified = approximate_polygon(contour, tolerance=tolerance)
        
        # Convert to integer coordinates for drawing
        r = np.round(simplified[:, 0]).astype(int)
        c = np.round(simplified[:, 1]).astype(int)
        
        # Ensure coordinates are within image bounds
        r = np.clip(r, 0, H-1)
        c = np.clip(c, 0, W-1)
        
        # Draw the simplified polygon
        if len(r) >= 3:  # Need at least 3 points to draw a polygon
            rr, cc = polygon(r, c, shape=(H, W))
            smoothed[rr, cc] = label
        else:
            # Fallback to original mask if simplified polygon is invalid
            smoothed[mask] = label
    
    # In case some pixels weren't assigned due to simplification
    # Assign any unassigned pixel to the majority label of its neighbors
    unassigned = (smoothed == 0) & (segments != 0)
    if np.any(unassigned):
        smoothed = fill_unassigned_pixels(smoothed, segments, unassigned)
    
    return smoothed



from skimage.measure import regionprops_table
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
def calculate_features(img, segments):
    features = {}
    #Statistical features - min, max, avg, std reflectance
    #0.2125 * R + 0.7154 * G + 0.0721 * B for grayscale reflectance
    #Spectral features - mean intensity per band (R, G, B, NIR), Standard deviation intensity per band
    #Indices - NDVI, NDWI, MSAVI2, SAVI
    # Texture - GLCM (contrast, entropy, homogeneity, correlation per band)
    # Shape - solidity, first 3 hu moments
    labels = ['R', 'G', 'B', 'NIR']
    for i, name in enumerate(labels):
        props_i = regionprops_table(
            segments,
            intensity_image=img[:, :, i],
            properties=['label', 'intensity_mean', 'intensity_std']
        )
        for k, v in props_i.items():
            if k == 'label':
                features.setdefault('segmentID', v)
            else:
                features[f'{name}_{k}'] = v
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    nir = img[:, :, 3]
    #Statistical stats
    gray_reflectance = 0.2125 * red + 0.7154 * green + 0.0721 * blue
    gray_reflec_props = regionprops_table(
        segments,
        intensity_image=gray_reflectance,
        properties=['intensity_mean', 'intensity_std', 'intensity_min', 'intensity_max']
    )
    for k, v in gray_reflec_props.items():
            features[f'reflec_{k}'] = v

    #indices
    indices = {
    'ndvi':(nir - red) / (nir + red + 1e-5),
    'ndwi':(green - nir) / (green + nir + 1e-5),
    'msavi2':(1/2)*(2*(nir+1)-np.sqrt((2*nir+1)**(2)-8*(nir-red))),
    'savi':((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    }
    
    for k, v in indices.items():
        props = regionprops_table(
            segments,
            intensity_image=v,
            properties=['intensity_mean']
        )
        features[k] = props['intensity_mean']

    # GLCM Texture features
    for i, name in enumerate(labels):
        for metric in ['contrast', 'homogeneity', 'correlation', 'entropy']:
            features[f'{name}_{metric}'] = []
    
    # Process each segment
    props = regionprops(segments)
    for region in props:
        minr, minc, maxr, maxc = region.bbox
        mask = np.zeros_like(segments, dtype=bool)
        mask[segments == region.label] = True
        
        # Process each band
        for i, name in enumerate(labels):
            img_band = img_as_ubyte(img[:, :, i])  # Scale to 0-255
            
            # Get the patch and apply mask
            region_mask = mask[minr:maxr, minc:maxc]
            region_img = img_band[minr:maxr, minc:maxc]
            
            # Check if region is large enough for texture calculation
            if np.sum(region_mask) <= 5:  # Minimum size threshold
                # Too small for meaningful texture
                for metric in ['contrast', 'homogeneity', 'correlation', 'entropy']:
                    features[f'{name}_{metric}'].append(np.nan)
                continue
                
            # Create a cropped image containing only the region pixels
            masked_img = np.zeros_like(region_img)
            masked_img[region_mask] = region_img[region_mask]
            
            # Check if region has sufficient intensity variation
            if np.unique(masked_img[region_mask]).size <= 1:
                # No texture information if all pixels have same value
                for metric in ['contrast', 'homogeneity', 'correlation', 'entropy']:
                    features[f'{name}_{metric}'].append(0.0)  # Use 0 for no texture
                continue
                
            # Calculate GLCM - use multiple angles and distances for robustness
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees
            glcm = graycomatrix(masked_img, distances=distances, angles=angles, 
                               levels=256, symmetric=True, normed=True)
            
            # Calculate properties and average over all angles
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))
            
            # Calculate entropy using graycoprops if available, otherwise manually
            # Some scikit-image versions don't have entropy in graycoprops
            try:
                entropy = np.mean(graycoprops(glcm, 'entropy'))
            except:
                # Manual entropy calculation
                entropy = -np.mean(np.sum(glcm * np.log2(glcm + 1e-10), axis=(0, 1)))
            
            # Store the features
            features[f'{name}_contrast'].append(contrast)
            features[f'{name}_homogeneity'].append(homogeneity)
            features[f'{name}_correlation'].append(correlation)
            features[f'{name}_entropy'].append(entropy)
    
    #Shape information
    shape_props = regionprops_table(
        segments,
        properties = ['solidity', 'moments_hu']
    )
    for prop in ['solidity', 'moments_hu-0', 'moments_hu-1', 'moments_hu-2']:
        features[prop]=shape_props[prop]

    return pd.DataFrame(features)



from shapely.geometry import shape
from rasterio.features import shapes
from skimage.segmentation import relabel_sequential
def feature_extraction(img, segments, transform, crs, features):
    # Convert segments to vector polygons
    polygons = [
        {
            'geometry': shape(geom),
            'properties': {'segmentID': int(val)}
        }
        for geom, val in shapes(segments.astype(np.int32), transform=transform)
        if val != 0  # Skip background if needed
    ]
    gdf = gpd.GeoDataFrame.from_features(polygons, crs = crs)
    gdf = gdf.dissolve(by='segmentID', as_index=False)
    gdf = gdf.merge(features, on='segmentID')
    
    return gdf


def process_all_rasters(stations, data_loc):
    def process_segment_extract(gdf_row, system, loc):
        try:
            name = re.sub(r'[^a-zA-Z0-9]', '_', gdf_row['stop_name'])[:30]
            raster = f"NAIP_{system}_{name}_400m.tif"
            
            station_dir=os.path.join(loc+'/'+system, name)
            if os.path.exists(station_dir):
                print(name, 'already exists')
                return station_dir
            print(f'Processing {name}')
            img, transform, crs, img_path = preprocess_raster(gdf_row, system, loc)
            corr_img = np.transpose(img, (1, 2, 0))

            print(f'Segmenting {name}')
            segments = object_segmentation(img[:3], fs_sigma=1, fs_scale=100,fs_min_size=200)
            segments = merge_small_segments(segments, 400)
            segments = merge_small_segments(segments, 700)

            print(f'Smoothing boundaries for {name}')
            segments = smooth_segment_boundaries(segments, 4)

            segments, _, _ = relabel_sequential(segments)

            print(f'Extracting features for {name}')
            features = calculate_features(corr_img, segments)
            poly = feature_extraction(img, segments, transform, crs, features)

            print(f'Writing data for {name}')
            
            os.makedirs(station_dir, exist_ok=True)
            new_img_path = os.path.join(station_dir, raster)
            shutil.move(img_path, new_img_path)

            shp_path = os.path.join(station_dir, f"{name}.shp")
            poly.to_file(shp_path)
            return station_dir
        except Exception as e:
            print(f'Failed on {name}:', e)
            return None
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for system, gdf in stations.items():
            for _, station_row in gdf.iterrows():
                futures.append(
                    executor.submit(
                        process_segment_extract,
                        station_row,
                        system,
                        data_loc
                    )
                )
        
        # Wait for all tasks to complete
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    print(f"Processed {len([r for r in results if r])} stations successfully")
    return results


stations = bp_sq25_data_acq.get_coordinates()
# print(stations['MTS'])

# img, transform, crs = preprocess_raster(stations['LAM'].iloc[1,:], 'LAM', '/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/Rasters/')
# corr_img = np.transpose(img, (1, 2, 0))

# segments = object_segmentation(img[:3], fs_sigma=1, fs_scale=100,fs_min_size=200)
# segments = merge_small_segments(segments, 400)
# segments = merge_small_segments(segments, 700)

# # segments = smooth_segment_boundaries(segments, 4)

# segments, _, _ = relabel_sequential(segments)

# features = calculate_features(corr_img, segments)
# poly = feature_extraction(img, segments, transform, crs, features)
# plt.figure(figsize=(12, 10))
# plt.imshow(img[:3].transpose(1, 2, 0))
# # plt.imshow(segments, cmap='tab20', alpha=0.3, interpolation='none')
# # boundaries = find_boundaries(segments, mode='thick')
# # plt.contour(boundaries, colors='black', linewidths=0.5)
# plt.axis('off')
# plt.title(f'Image with {len(np.unique(segments))} segments')
# plt.tight_layout()
# plt.show()

# process_segment_extract(stations['LAM'].iloc[1,:], 'LAM', '/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/Rasters/')
subset = {
    'MTS': stations['MTS'],
    'LAM': stations['LAM'],
    'VTA': stations['VTA']
}
process_all_rasters(subset, '/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/Rasters/')