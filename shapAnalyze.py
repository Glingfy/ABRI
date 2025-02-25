#!/usr/bin/env python
# -*- coding:utf-8 -*-

import shap
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def save_plot(figure, save_path, file_name, dpi=300, format='jpg'):
    """
    Save a matplotlib figure to a specified path.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)          # Create the directory if it doesn't exist
    full_path = os.path.join(save_path, f"{file_name}.{format}")
    figure.savefig(full_path, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Saved: {full_path}")


# load model
model_path = r'document/model.m'
bst = joblib.load(model_path)

# Load data
filePath = r'path1/documents/file.xlsx'
data = pd.read_excel(filePath, sheet_name='Sheet')
inputs = ['feature1', 'feature2', 'feature3', '...']        # List of inputs, include TN, TP, RNP, DO, T and precipitation
target = 'name'     # name of target variable
X = data[inputs]
y = data[target]

out_path = r'path2/documents'
# Ensure output path exists
if not os.path.exists(out_path):
    os.makedirs(out_path)

# SHAP analysis
explanier = shap.TreeExplainer(bst)
shap_values = explanier.shap_values(X)

# summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
save_plot(plt.gcf(), out_path, 'summary')

# Bar plot: take the average of the absolute value of each feature's SHAP value as the importance of that feature
plt.figure()
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
save_plot(plt.gcf(), out_path, 'bar')

# Dependence plots: depict the interaction between two features, using T and precipitation as an example
plt.figure()
shap.dependence_plot('T', shap_values, X, interaction_index='precipitation', show=False)
save_plot(plt.gcf(), out_path, 'T_precipitation')

# Output prediction results
y_pred = bst.predict(X)
y_df = pd.DataFrame({'actual': y, 'predict': y_pred})
shapValue = pd.DataFrame(shap_values, columns=[f"{col}_shap" for col in X.columns])
shapValue = shapValue.join(X)

# Save results to Excel
output_file = os.path.join(out_path, 'SHAP.xlsx')
with pd.ExcelWriter(output_file) as writer:
    y_df.to_excel(writer, sheet_name='model', index=False)
    shapValue.to_excel(writer, sheet_name='ShapValue', index=False)
print(f"Saved prediction and SHAP values to: {output_file}")

