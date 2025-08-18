import pandas as pd
import geopandas as gpd
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import bp_sq25_data_acq

train_shapefiles = {
    'LAM': ['Union_Station___Metro_A_Line', 'Maravilla_Station', 'Redondo_Beach_Station', 'Crenshaw_Station',
            'Culver_City_Station', 'APU___Citrus_College_Station', 'Duarte___City_of_Hope_Station', 'Vernon_Station'],
    'MTS': ['Gaslamp_Quarter_Station', 'SDSU_Station', 'Stadium_Station', 'Clairemont_Drive_Station'],
    'sacRT': ['Sacramento_Valley_Station', 'Archives_Plaza_Station', 'Historic_Folsom_Station'],
    'SFMTA': ['Chinatown___Rose_Pak_Station_', '4th_St___King_St'],
    'VTA': ['Great_America', 'Saint_James']
}
paths = []
pth = '/Users/jeremyelvander/Desktop/AISC_BP_SQ25/Data/Rasters/'
for system, stations in train_shapefiles.items():
    sys_path = os.path.join(pth, system)
    for station in stations:
        station_path = os.path.join(sys_path, station)
        shp = glob.glob(os.path.join(station_path, '*.shp'))
        paths.extend(shp)

data = []
for shp in paths:
    gdf = gpd.read_file(shp)
    gdf = gdf.drop(columns=['segmentID', 'geometry', 'class'], errors='ignore')

    if 'LULC' not in gdf.columns:
        continue
    labeled = gdf.dropna(subset=['LULC'])

    if not labeled.empty:
        data.append(labeled)

training_data = pd.concat(data, ignore_index=True)
training_data['LULC'] = training_data['LULC'].replace('high_density_res', 'urban')
training_data['LULC'] = training_data['LULC'].replace('institutional', 'urban')
training_data['LULC'] = training_data['LULC'].replace('industrial', 'urban')
training_data['LULC'] = training_data['LULC'].replace('low_density_res', 'urban')
training_data['LULC'] = training_data['LULC'].replace('commercial', 'urban')
training_data['LULC'] = training_data['LULC'].replace('agriculture', 'undeveloped')


X = training_data.iloc[:, 0:35]
le = LabelEncoder()
y = le.fit_transform(training_data['LULC'])

x_train, x_test, y_train, y_test= train_test_split(
    X, y, 
    test_size=0.1, 
    stratify=y, 
    random_state=42
)

# === Step 2: Define model and hyperparameter grid ===
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    'max_features': ['sqrt']
}

param_grid2 = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# === Step 3: Grid Search with Cross-Validation ===
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(x_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5)
print("Cross-validated accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

for i, class_name in enumerate(le.classes_):
    print(f"{i}: {class_name}")

y_pred = best_model.predict(x_test)
print(classification_report(y_test, y_pred))

# joblib.dump(grid, 'LULC_satellite_RF_classifier.joblib')

importances = best_model.feature_importances_
feature_names = X.columns

# import matplotlib.pyplot as plt
# import numpy as np
# # Sort features by importance
# indices = np.argsort(importances)[::-1]
# top_n = 20  # change as needed
# plt.figure(figsize=(10, 6))
# plt.title("Top Feature Importances")
# plt.bar(range(top_n), importances[indices[:top_n]], align="center")
# plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
# plt.tight_layout()
# plt.show()

# station_frame = bp_sq25_data_acq.get_coordinates()
# print(station_frame['MTS'])

# for system in train_shapefiles.keys():
#     sys_path = os.path.join(pth, system)
#     subdir = [station for station in os.listdir(sys_path)]
#     for station in subdir:
#         if station == '.DS_Store':
#             continue
#         print(f'predicting {system} - {station}')
#         station_path = os.path.join(sys_path, station)
#         shp = glob.glob(os.path.join(station_path, '*.shp'))
#         gdf = gpd.read_file(shp[0])
#         gdf=gdf.drop(columns=['class'], errors='ignore')
#         gdf_features = gdf.drop(columns=['segmentID', 'geometry', 'LULC'], errors='ignore')
#         predictions = best_model.predict(gdf_features.iloc[:, :X.shape[1]])
#         predicted_labels = le.inverse_transform(predictions)

#         # Add predictions as new column
#         gdf['class'] = predicted_labels

#         gdf.to_file(shp[0])

