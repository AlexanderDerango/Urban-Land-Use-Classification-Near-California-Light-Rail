# Urban Land Use Classification Near California Light Rail

A geospatial machine learning project that uses high-resolution NAIP aerial imagery to classify land use around California light rail stations. The project combines GIS, object-based image analysis, remote sensing features, and Random Forest classification to analyze land use patterns across five California transit systems.

## About

The goal of this project was to explore how machine learning and geospatial data can be used to understand land use around public transportation.

We analyzed light rail stations across five California transit systems:

* Sacramento Regional Transit (sacRT)
* San Francisco Muni Metro (SFMTA)
* Valley Transportation Authority (VTA)
* Los Angeles Metro (LAM)
* San Diego Metropolitan Transit System (MTS)

For each station, we created a 400-meter buffer and analyzed high-resolution NAIP aerial imagery to classify surrounding land into five categories:

* Buildings
* Transportation
* Parks / Open Space
* Water
* Undeveloped Land

## Project Workflow

### 1. Data Acquisition

Station coordinates were collected using GTFS transit feeds, web scraping, and XML parsing. NAIP imagery was retrieved through Microsoft's Planetary Computer.

The imagery had a resolution of approximately 0.6 meters and was reprojected into appropriate projected coordinate systems for spatial analysis.

### 2. Image Segmentation

Images were segmented into meaningful objects for **Object-Based Image Analysis (OBIA)**.

The project used the Felzenszwalb graph-based segmentation algorithm, followed by post-processing to smooth polygons, simplify boundaries, and merge small segments.

### 3. Feature Extraction

Remote sensing, statistical, textural, and geometric features were extracted from each image segment.

Features included:

* NDVI
* NDWI
* MSAVI2
* SAVI
* RGB and NIR reflectance statistics
* GLCM texture features
* Polygon shape features
* Hu moments
* Solidity

Approximately 2,000 segments were manually labeled across selected stations and used to train the classification model.

### 4. Machine Learning

A **Random Forest classifier** was trained to classify each segmented object into one of the five land-use categories.

GridSearchCV was used for hyperparameter tuning and cross-validation.

**Best parameters:**

* `n_estimators = 200`
* `max_depth = 30`
* `max_features = sqrt`
* `min_samples_leaf = 2`
* `min_samples_split = 2`

### 5. Land Use Analysis

After classifying the image segments, we calculated the proportion of each land-use category within the 400-meter station buffers.

This allowed us to compare surrounding land use across individual stations and transit systems.

## Results

The Random Forest model achieved approximately **82% cross-validation accuracy**.

Performance varied across land-use categories. The model performed particularly well for parks/open space, with a precision of approximately **0.91**, while water was more difficult to classify, with an F1-score of approximately **0.67**.

The model generally performed well at identifying larger, clearly defined objects such as buildings. Smaller segmentation artifacts and visual similarities between building shadows and water created additional classification challenges.

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

These results demonstrate how geospatial machine learning can be used to compare land-use patterns across large transit systems.

## Technologies

### Machine Learning & Computer Vision

* Python
* Scikit-learn
* Random Forest
* Object-Based Image Analysis (OBIA)
* Image segmentation
* Scikit-image

### GIS & Remote Sensing

* GeoPandas
* Rasterio
* Shapely
* QGIS
* NAIP aerial imagery
* Microsoft Planetary Computer
* NDVI / NDWI / SAVI / MSAVI2
* GLCM texture analysis

### Data Acquisition

* GTFS
* Web scraping
* XML parsing
* Requests
* lxml
* Fuzzy string matching

## Project Structure

```text
AISC-ML-Project-Spring2025/
├── bp_sq25_data_acq.py
├── land_use_mix.py
├── modeling.py
├── segmentation_features.py
├── requirements.txt
└── README.md
```

* `bp_sq25_data_acq.py` — Collects station coordinates and NAIP imagery
* `land_use_mix.py` — Performs image segmentation and processing
* `modeling.py` — Trains and applies the Random Forest classifier
* `segmentation_features.py` — Extracts features and calculates land-use proportions

## Future Work

Potential improvements include:

* Improving image segmentation accuracy
* Developing better shadow-removal techniques
* Experimenting further with deep learning segmentation models
* Comparing land use with station ridership
* Analyzing additional stations across each transit system
* Incorporating additional imagery or geospatial datasets
* Improving classification of smaller or visually similar objects

## Takeaways

This project provided hands-on experience combining machine learning with GIS and remote sensing. A major takeaway was that model performance depends heavily on the quality of the underlying image segmentation and feature engineering.

The project also demonstrated how geospatial machine learning can be applied beyond individual image classification to conduct larger-scale comparisons of land use across transportation systems.
