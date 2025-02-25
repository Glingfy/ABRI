# Introduction

This repository contains the data and partial code used in the article titled "A novel algal bloom risk assessment framework by integrating environmental factors based on explainable machine learning."

## 1. Data File: `field_measurements.xlsx`

This file includes three sheets: `rivers`, `lakes`, and `stations`.

- **`stations` sheet**: Contains three fields: station `ID` and geographic coordinates (`latitude` and `longitude`).
- **`rivers` and `lakes` sheets**: Include water quality data from monitoring sites for rivers and lakes (including reservoirs), respectively. These sheets consist of 14 fields, including:
  - `ID`: Station identification number.
  - `Date`: Date of water quality measurement.
  - `T` (Water Temperature), `pH`, `DO` (Dissolved Oxygen), `Turbidity`, `Conductivity`, `CODMn` (Permanganate Index), `NH3N` (Ammonia-Nitrogen) , `TP` (Total Phosphorus), `TN` (Total Nitrogen), `Chla` (Chlorophyll a): Water quality parameters obtained from the monitoring sites.
  - `Type`: Type of water body being monitored (e.g., river, lake).

These datasets were derived from raw monitoring data after undergoing a data cleaning process.

## 2. Code: `buildML.py`

This script is designed to construct three types of machine learning models: Random Forest, Support Vector Regression (SVR), and XGBoost. The models are optimized using Bayesian optimization techniques. Model performance is evaluated using three metrics: Concordance Correlation Coefficient (CCC), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

- Depending on the modeling objective, different input features should be selected. For example:
  - When constructing a chlorophyll-a estimation model based on Sentinel-2 MSI data, input features should include combinations of reflectance bands from Sentinel-2.
  - When building a model to assess the factors influencing algal blooms, input features should include key environmental factors such as Total Nitrogen, Total Phosphorus, and Dissolved Oxygen.

## 3. Code: `shapML.py`

This script performs SHAP (SHapley Additive exPlanations) analysis based on a trained XGBoost model to interpret the contributions of each feature to the model's predictions.

## 4. Code: `ABRI.py`

This script is used to calculate Algal Bloom Risk Index (ABRI) for raster images (.tif). 
