# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:56:46 2023

@author: esa
"""
import pcraster as pcr
import numpy as np
import xarray as xr
import pandas as pd
import subprocess
import os
import glob
from osgeo import gdal, osr, ogr


import UserInputs as ui

def open_nc(nc, chunks):
    with xr.open_dataset(nc, chunks=chunks) as dts:
        key=list(dts.keys())[0]
        var=dts[key]
        dts.close()
        return var,key

def GetGeoInfo(fh, subdataset = 0):
    """
    Substract metadata from a geotiff, HDF4 or netCDF file.
    
    Parameters
    ----------
    fh : str
        Filehandle to file to be scrutinized.
    subdataset : int, optional
        Layer to be used in case of HDF4 or netCDF format, default is 0.
        
    Returns
    -------
    driver : str
        Driver of the fh.
    NDV : float
        No-data-value of the fh.
    xsize : int
        Amount of pixels in x direction.
    ysize : int
        Amount of pixels in y direction.
    GeoT : list
        List with geotransform values.
    Projection : str
        Projection of fh.
    """
    SourceDS = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = SourceDS.GetDriver().ShortName
    if Type == 'HDF4' or Type == 'netCDF':
        SourceDS = gdal.Open(SourceDS.GetSubDatasets()[subdataset][0])
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    driver = gdal.GetDriverByName(Type)
    return driver, NDV, xsize, ysize, GeoT, Projection

def convert_tif_to_pcr(map2convert_fh, dtype):
    if "Float" in dtype:
        pcr_type = 'VS_SCALAR'
    if "Int" in dtype:
        pcr_type = 'VS_NOMINAL'
    output_fh = map2convert_fh[:-4]+'.map'
    fullCmd = 'gdal_translate -ot %s -of PCRaster -mo PCRASTER_VALUESCALE=%s %s %s' %(dtype, pcr_type, map2convert_fh, output_fh)
    subprocess.Popen(fullCmd)
    return output_fh

def convert_pcr_to_tif(map2convert_fh):
    output_fh = map2convert_fh[:-4]+'.tif'
    fullCmd = 'gdal_translate -a_srs EPSG:4326 %s %s' %(map2convert_fh, output_fh)
    subprocess.Popen(fullCmd)

def OpenAsArray(fh, bandnumber = 1, dtype = 'float32', nan_values = False):
    """
    Open a map as an numpy array. 
    
    Parameters
    ----------
    fh: str
        Filehandle to map to open.
    bandnumber : int, optional 
        Band or layer to open as array, default is 1.
    dtype : str, optional
        Datatype of output array, default is 'float32'.
    nan_values : boolean, optional
        Convert he no-data-values into np.nan values, note that dtype needs to
        be a float if True. Default is False.
        
    Returns
    -------
    Array : ndarray
        Array with the pixel values.
    """
    datatypes = {"uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16":  np.int16, "Int16":  np.int16, "uint32": np.uint32,
    "int32": np.int32, "float32": np.float32, "float64": np.float64, "complex64": np.complex64, "complex128": np.complex128,
    "Int32": np.int32, "Float32": np.float32, "Float64": np.float64, "Complex64": np.complex64, "Complex128": np.complex128,}
    DataSet = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = DataSet.GetDriver().ShortName
    if Type == 'HDF4':
        Subdataset = gdal.Open(DataSet.GetSubDatasets()[bandnumber][0])
        NDV = int(Subdataset.GetMetadata()['_FillValue'])
    else:
        Subdataset = DataSet.GetRasterBand(bandnumber)
        NDV = Subdataset.GetNoDataValue()
    Array = Subdataset.ReadAsArray().astype(datatypes[dtype])
    if nan_values:
        Array[Array == NDV] = np.nan
    return Array


wdir = r'E:\Master\Master_thesis\WetSpa-Python_v1_0'

fhs = glob.glob(os.path.join(r'E:\Master\Master_thesis\WetSpa-Python_v1_0\output\Preprocess_maps','*.map'))
for fh in fhs:
    convert_pcr_to_tif(fh)

watershed = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','watershed.tif'))  
t0_h = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','t0_h.tif'))
delta_h = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','delta_h.tif'))
     
nsub = int(np.nanmax(watershed))
maxt = np.nanmax(t0_h+5*(delta_h))
mt = 1+int(0.5+maxt)
if (mt<1):
    print('Error: maximum size of IUH-vector less than 1!')

mint = t0_h-3*(delta_h) # minimum length of the iuh
mint = int(np.nanmin(np.where(mint<0, 0, mint)))
maxt = np.nanmin(np.where(maxt>mt, mt, maxt)) # maximum length of the iuh
maxt = int(np.where(maxt<1, 1, maxt))

print ('number of sub-catchment of your area: ', nsub)
print ('maxt= ', maxt)
print('mint= ', mint)
print ('mt= ', mt)



























