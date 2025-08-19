# AISC-ML-Project-Spring2025

## Image Analysis for Urban Land Use Classification Near California Light Rail

This project processes NAIP imagery for multiple transit stops in California to analyze land use around each station. Land is classified into urban, transportation, parks, water, or undeveloped categories using a machine learning model. Transit stops include light rail in Sacramento, San Francisco, San Jose, San Diego, and Los Angeles. 

### Project Workflow
 * Data Acquisition:
   * Gather coordinate data for all California light rail stations via GTFS feeds and web scraping
   * Retrieve NAIP imagery from Microsoft Planetary Computer
 * Segmentation
   * Perform graph-based image segmentation
   * Post-process segments for Object-Based Image Analysis
 * Feature Extraction & Modeling
   * Extract remote sensing features (e.g., NDVI, NDWI, GLCM texture).
   * Train a Random Forest model to classify land use categories.
 * Land Use Mix Calculation
   * Create 400m station buffers.
   * Calculate proportions of land use types for each station.

### Project Structure
 * bp_sq25_data_acq.py - Collects station coordinates & NAIP imagery
 * land_use_mix.py - Performs image segmentation & processing
 * modeling.py - Builds & applies Random Forest classifier
 * segmentation_features.py - Aggregates results & calculates land use proportions 
 * requirements.txt - Python dependencies

### Instructions for Running Program
 * Run in virtual environment
 * Install packages
 * Run the files in Project Structure in order

### Contact Information
Email: derangoalexander@gmail.com
