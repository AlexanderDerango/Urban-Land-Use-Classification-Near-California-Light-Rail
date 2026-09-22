# Urban Land Use Classification Near California Light Rail

## About

A machine learning and GIS project that uses high-resolution NAIP aerial imagery to classify land use around California light rail stations. The project combines image segmentation, remote sensing features, and Random Forest classification to analyze land-use patterns across five transit systems.

## Project Overview

The analysis covers five California light rail systems:

* Sacramento Regional Transit (sacRT)
* San Francisco Muni Metro (SFMTA)
* Valley Transportation Authority (VTA)
* Los Angeles Metro (LAM)
* San Diego Metropolitan Transit System (MTS)

The final classification uses five land-use classes:

* Urban
* Transportation
* Undeveloped
* Parks/Open Space
* Water

Each station was analyzed using a **400-meter buffer** around the station.

## Methodology

### 1. Station & Imagery Data

Station locations were collected from transit agency datasets and web sources and converted to appropriate projected coordinate systems. NAIP aerial imagery was retrieved through the **Microsoft Planetary Computer**. Multiple imagery tiles were mosaicked when a station area crossed tile boundaries.

### 2. Image Segmentation

NAIP imagery was normalized and preprocessed before applying the **Felzenszwalb graph-based segmentation algorithm**. Small segments were merged and boundaries were smoothed to create geographic objects for analysis.

### 3. Feature Extraction

Features were calculated for each segmented object, including:

* RGB and NIR reflectance statistics
* NDVI, NDWI, MSAVI2, and SAVI
* GLCM texture features
* Shape features including solidity and Hu moments

The resulting segments and features were converted into geographic polygons for classification and analysis.

### 4. Machine Learning

A **Random Forest classifier** was trained using manually labeled land-use segments. `GridSearchCV` was used to select model parameters, and performance was evaluated using 5-fold cross-validation and a held-out test set.

The final model achieved approximately **82% cross-validation accuracy**. Classification performance varied by land-use class, with water being more difficult to distinguish from other classes.

### Example Land Use Analysis

The classified imagery was used to estimate land-use proportions around individual stations and compare the five transit systems.

For example, the analysis included:

| Transit System | Urban | Transportation | Undeveloped | Parks | Water |
| -------------- | ----: | -------------: | ----------: | ----: | ----: |
| SFMTA          | 64.4% |          25.6% |        1.9% |  7.0% |  1.0% |
| sacRT          | 51.8% |          27.5% |        5.3% | 14.7% |  0.7% |
| VTA            | 54.2% |          30.3% |        4.1% | 11.0% |  0.4% |
| LAM            | 60.6% |          30.2% |        3.0% |  5.2% |  0.9% |
| MTS            | 50.8% |          31.7% |        4.8% | 11.3% |  1.5% |

<img width="567" height="557" alt="image" src="https://github.com/user-attachments/assets/05f5aad6-121d-4852-99ba-abd1b6f32c90" />

* Classified Image - Archives Plaza Station, sacRT

<img width="357" height="332" alt="image" src="https://github.com/user-attachments/assets/2f33362e-3c9d-49c3-b6e0-f3ec4ce5d652" />

* Segmented Image - Archives Plaza Station, sacRT

These results demonstrate how geospatial machine learning can be used to compare land-use patterns across large transit systems.

## Technologies

**Python · GeoPandas · Rasterio · scikit-image · scikit-learn · Shapely · Pandas · NumPy · OpenCV · QGIS · Microsoft Planetary Computer**

## Project Structure

```text
├── bp_sq25_data_acq.py
├── segmentation_features.py
├── modeling.py
├── land_use_mix.py
└── requirements.txt
```

## Future Work

* Improve segmentation and boundary quality
* Improve classification of visually similar land-use classes
* Expand the analysis to additional stations
* Explore deep learning-based segmentation and classification
* Compare land-use patterns with transit ridership
* Incorporate additional geospatial datasets

The project also demonstrated how geospatial machine learning can be applied beyond individual image classification to conduct larger-scale comparisons of land use across transportation systems.
