#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Calculation of ABRI based on Tiff image
"""
import os
from osgeo import gdal
import numpy as np
import pandas as pd
from datetime import datetime


def lakeABRI(df: pd.DataFrame):
    """
    ABRI for lakes
    """
    df['ci'] = df['chla'].apply(lambda x: 0.04 * x if x < 25 else 1)
    df['pi'] = df['TP'].apply(lambda x: 437.4 * x - 4.76 if x < 0.018 else 3.11)
    df['ni'] = df['TN'] * 0.99 - 2.78
    df['RNP'] = df['TN'] / df['TP']
    df['npi'] = df['RNP'].apply(lambda x: -0.025 * x + 1.82 if x < 180 else -0.88)
    df['oi'] = df['DO'].apply(lambda x: 0.66 * x - 5.22 if x < 10.07 else -1.55 * x + 17.03)
    df['ti'] = df['T'].apply(lambda x: 0.42 * x - 5.86 if x < 18.1 else -0.27 * x + 6.63)
    df['prei'] = df['precipitation'].apply(lambda x: -2.49 * x + 0.06 if x < 0.14 else -0.29)
    df['risk2'] = df['pi'] + df['ni'] + df['npi'] + df['oi'] + df['ti'] + df['prei']
    df['risk'] = df['ci'] * 0.5 + (df['risk2'] + 2.4) / (5.3 + 10.2)
    return df['risk']


def riverABRI(df: pd.DataFrame):
    """
    ABRI for rivers
    """
    df['ci'] = df['chla'].apply(lambda x: 0.025 * x if x < 40 else 1)
    df['pi'] = df['TP'].apply(lambda x: 89.62 * x - 6.23 if x < 0.124 else (7.47 if x > 0.227 else 25.06 * x + 1.78))
    df['ni'] = df['TN'] * 0.99 - 2.78
    df['RNP'] = df['TN'] / df['TP']
    df['npi'] = df['RNP'].apply(lambda x: -0.09 * x + 3.42 if x < 41.8 else -0.342)
    df['oi'] = df['DO'].apply(lambda x: 0.85 * x - 10.14 if x < 8 else (16.61 if x > 15 else 2.85 * x - 26.14))
    df['ti'] = df['T'] * 0.56 - 10.89
    df['prei'] = df['precipitation'].apply(lambda x: -2.49 * x + 0.06 if x < 0.14 else -0.29)
    df['risk2'] = df['pi'] + df['ni'] + df['npi'] + df['oi'] + df['ti'] + df['prei']
    df['risk'] = df['ci'] * 0.5 + (df['risk2'] + 0.3) / (34.4 + 16.3)
    return df['risk']


def readData(fileList: list, nameOrder: list):
    """
    Read multiple images and convert them to dataframe
    :param fileList: Input image list
    :param nameOrder: Image name list
    :return: Image data, shape, projection information
    """
    allData = []
    shape = None
    geotransform = None
    projection = None
    for file in fileList:
        data_i = gdal.Open(file)
        if data_i is None:
            raise Exception(f"Unable to open file:{file}")
        band_i = data_i.GetRasterBand(1)
        array_i = band_i.ReadAsArray().astype(np.float16)
        shape = array_i.shape
        geotransform = data_i.GetGeoTransform()
        projection = data_i.GetProjectionRef()
        allData.append(array_i.flatten(order='C'))
    allData = np.array(allData, dtype=np.float32)
    print('---Image data was read successfully---')
    return pd.DataFrame(allData.T, columns=nameOrder), shape, geotransform, projection


def readDataBlock(fileList: list, nameOrder: list, block_size: tuple):
    """
    Read multiple images in blocks and convert them to dataframe
    :param fileList: Input image list
    :param nameOrder: Image name list
    :param block_size: Block size for reading (e.g., (256, 256))
    :return: Image data, shape, projection information
    """
    allData = []
    shape = None
    geotransform = None
    projection = None

    for file in fileList:
        data_i = gdal.Open(file)
        if data_i is None:
            raise Exception(f"Unable to open file: {file}")

        band_i = data_i.GetRasterBand(1)
        shape = (data_i.RasterYSize, data_i.RasterXSize)
        geotransform = data_i.GetGeoTransform()
        projection = data_i.GetProjectionRef()

        # Initialize an empty array for the entire image
        array_i = np.full(shape, np.nan, dtype=np.float32)

        # Read data in blocks
        for i in range(0, shape[0], block_size[0]):
            for j in range(0, shape[1], block_size[1]):
                width = min(block_size[1], shape[1] - j)
                height = min(block_size[0], shape[0] - i)
                array_i[i:i + height, j:j + width] = band_i.ReadAsArray(j, i, width, height).astype(np.float32)

        allData.append(array_i.flatten(order='C'))

    allData = np.array(allData, dtype=np.float32)
    print('---Image data read successfully---')
    return pd.DataFrame(allData.T, columns=nameOrder), shape, geotransform, projection


def writetif(im_bands, geotransform, projection, img, outimgname, no_data_value=None):
    """
    Output the result as tiff
    """
    options = ['COMPRESS=LZW', 'TILED=YES']
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outimgname, img.shape[1], img.shape[0], im_bands, gdal.GDT_Float32, options=options)
    dataset.SetProjection(projection)       # Write projection
    dataset.SetGeoTransform(geotransform)       # Writes affine transformation parameters
    if no_data_value is not None:
        for i in range(im_bands):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(no_data_value)
            band.WriteArray(img[i] if im_bands > 1 else img)
            band.FlushCache()
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  # Write array data
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset


# Examples
if __name__ == "__main__":
    # Read reference images to distinguish bodies of water from rivers
    water_reference = r'waterImage/water.tif'
    ds = gdal.Open(water_reference, gdal.GA_ReadOnly)
    if ds is None:
        raise Exception(f"Unable to open file:{water_reference}")
    reference = ds.GetRasterBand(1).ReadAsArray()
    shape = reference.shape
    reference = reference.flatten(order='C')

    # Load data
    file_path = r'path/document'
    factors = ['TP', 'TN', 'DO', 'T', 'precipitation', 'chla']
    file_list = [os.path.join(file_path, f"{factor}.tif") for factor in factors]
    print("File list:", file_list)
    data, shape, geotransform, projection = readData(file_list, factors)

    # If the amount of image data to be processed is large, the image can be read in blocks
    block_size = (256, 256)  # Define block size for reading large files, and adjust block size as needed
    data, shape, geotransform, projection = readDataBlock(file_list, factors, block_size)

    no_data_value = 9999.0
    data['reference'] = reference   # Filter water bodies (lake=1, river=2)
    data['risk'] = no_data_value
    water = data.loc[data['chla'] > 0].copy()  # Select water bodies with valid chla values
    water_index = water.index

    # Handle missing and infinite values
    water.replace([np.inf, -np.inf], np.nan, inplace=True)
    water.fillna(water.median(), inplace=True)

    # Calculate risk for rivers and lakes
    river = water.loc[water['reference'] == 2].copy()
    lake = water.loc[water['reference'] == 1].copy()

    if not river.empty:
        river['risk'] = riverABRI(river)
        water.loc[river.index, 'risk'] = river['risk']

    if not lake.empty:
        lake['risk'] = lakeABRI(lake)
        water.loc[lake.index, 'risk'] = lake['risk']

    # Update main dataframe
    data.loc[water_index, 'risk'] = water['risk']

    # save result tiff
    risk = data['risk'].to_numpy().reshape(shape)
    out_name = r'path2/document/ABRI.tif'
    writetif(1, geotransform, projection, risk, out_name, no_data_value)

    # Clear cache
    del data, water, river, lake, risk
