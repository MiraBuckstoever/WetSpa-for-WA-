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

def CreateGeoTiff(fh, Array, driver, NDV, xsize, ysize, GeoT, Projection, explicit = True, compress = None):
    """
    Creates a geotiff from a numpy array.
    
    Parameters
    ----------
    fh : str
        Filehandle for output.
    Array: ndarray
        Array to convert to geotiff.
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
    datatypes = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 3, "Int16": 3, "uint32": 4,
    "int32": 5, "float32": 6, "float64": 7, "complex64": 10, "complex128": 11,
    "Int32": 5, "Float32": 6, "Float64": 7, "Complex64": 10, "Complex128": 11,}
    if compress != None:
        DataSet = driver.Create(fh,xsize,ysize,1,datatypes[Array.dtype.name], ['COMPRESS={0}'.format(compress)])
    else:
        DataSet = driver.Create(fh,xsize,ysize,1,datatypes[Array.dtype.name])
    if NDV is None:
        NDV = -9999
    if explicit:
        Array[np.isnan(Array)] = NDV
    DataSet.GetRasterBand(1).SetNoDataValue(NDV)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection.ExportToWkt())
    DataSet.GetRasterBand(1).WriteArray(Array)
    DataSet = None
    if "nt" not in Array.dtype.name:
        Array[Array == NDV] = np.nan


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

def nc2GeoTiff(nc_fh, template_fh):
    latchunk = 200
    lonchunk = 200
    timechunk = 1
    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk} 
    dataset, _ = open_nc(nc_fh, chunks)
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(template_fh)
    for t in range(len(dataset['time'])):
        data = dataset.isel(time=t).values
        f_name = nc_fh[:-3]+'_'+str(data.time.values)[:10]+'.tif'
        CreateGeoTiff(f_name,data, driver, NDV, xsize, ysize, GeoT, Projection)

def convert_tif_to_pcr(map2convert_fh, dtype):
    if "Float" in dtype:
        pcr_type = 'VS_SCALAR'
    if "Int" in dtype:
        pcr_type = 'VS_NOMINAL'
    output_fh = map2convert_fh[:-4]+'.map'
    fullCmd = 'gdal_translate -ot %s -of PCRaster -mo PCRASTER_VALUESCALE=%s %s %s' %(dtype, pcr_type, map2convert_fh, output_fh)
    subprocess.Popen(fullCmd)
    return output_fh

def ConvertToPCRaster(src_filename,dst_filename,ot,VS):
    #Open existing dataset
    src_ds = gdal.Open(src_filename)
    
    #GDAL Translate
    dst_ds = gdal.Translate(dst_filename, src_ds, format='PCRaster', outputType=ot, metadataOptions=VS)
    
    #Properly close the datasets to flush to disk
    dst_ds = None
    src_ds = None

def convert_pcr_to_tif(map2convert_fh, EPSG):
    output_fh = map2convert_fh[:-4]+'.tif'
    src_ds = gdal.Open(map2convert_fh)
    translateoptions = "-a_srs EPSG:%s" %(EPSG)
    dst_ds = gdal.Translate(output_fh, src_ds, options=translateoptions)
    
    dst_ds = None
    src_ds = None
    #fullCmd = 'gdal_translate -a_srs EPSG:%s %s %s' %(EPSG, map2convert_fh, output_fh)
    #subprocess.Popen(fullCmd)
    return output_fh

def reclass_based_on_one_map(table, input_map):
    #it replaces the values in the map according to their corresponding values in the table
    parameter = pcr.lookupscalar(table,input_map)
    fname = table[0:-4].replace('tables','Preprocess_maps',1)
    fname = fname.replace('input', 'output',1)+'.map'
    pcr.report(parameter,fname)

def manning(mask, streamOrder, landuse, wdir):
    # default values
#    defaultMinManning = 0.025
#    defaultMaxManning = 0.055
    
    MinManning = ui.minManning
    MaxManning = ui.maxManning
                        
    option = ui.ManningOption

    if option == "1":
        # set manning coefficients
#        MinManning = defaultMinManning 
#        MaxManning = defaultMaxManning

        # create global variable
        global streamManning
            
        # search max and min streamorder
        maxStreamorder = pcr.areamaximum(streamOrder, mask)
        minStreamorder = pcr.areaminimum(streamOrder, mask)
                    
        # calculate the Manning coefficient from streamorder by linear interpolation
        # (order- minOrder)/(maxOrder - minOrder) = (manning - maxManning)/(maxManning - minManning)  -> manning
        temp1 = (pcr.scalar(streamOrder) - pcr.scalar(minStreamorder)) / (pcr.scalar(maxStreamorder) - pcr.scalar(minStreamorder))
        temp2 = MaxManning - MinManning
        temp3 = temp1 * temp2
        #print(str(maxManning))
        streamManning = pcr.scalar(MaxManning) - temp3
            
    elif option == "2":
        # use the default Manning coefficient
        table = open(os.path.join(wdir,'input','tables','manning.tbl'))
        tableContents = table.readlines()
        # take value of line 12 (0.035), which is croplands (why not permanent wetlands (0.05) ?)
        # I think is 0.35
        defaultManning =float((((tableContents[11]).split('\n'))[0].split('\t'))[1])
            
        # create streamManning
        streamManning = pcr.ifthen(pcr.defined(streamOrder),pcr.scalar(defaultManning))
            
    else:
        # set the default Manning coefficient         
        defaultManning = ui.ManningValue
        # create streamManning
        streamManning = pcr.ifthen(pcr.defined(streamOrder),pcr.scalar(defaultManning))

    # reclassify landuse map to Manning coefficients
    landuseReclass = pcr.lookupscalar(os.path.join(wdir,'input','tables','manning.tbl'), landuse)
    # Manning map creation
    manning_coeff = pcr.ifthenelse(pcr.defined(streamManning),streamManning,landuseReclass)
    pcr.report(manning_coeff, os.path.join(wdir,'output','Preprocess_maps','Manning.map'))

    return manning_coeff

def runoff_coeff(slopemap, soil, landuse, mask, total_mask, wdir):

#    print('')
#    print('')
#    print('------ RUNOFF COEFFICIENT calculation ------')

    # Combine LANDUSE map (17 classes) with the SOIL map (12 classes) to calculate POTENTIAL RUNOFF and SLOPE_0 maps
    runoff_c0 = pcr.lookupscalar(os.path.join(wdir,'input','tables','runoff_coeff.tbl'), landuse, soil)
#    pcr.report(runoff_c0, os.path.join(wdir,'output','Preprocess_maps','runoff_co_0.map'))
    slope_0 = pcr.lookupscalar(os.path.join(wdir,'input','tables','slope0.tbl'), landuse, soil)
#    pcr.report(slope_0, os.path.join(wdir,'output','Preprocess_maps','slope_0.map'))
    
    # Calculate temporary runoff coefficients
    rogrid1 = (1 - runoff_c0) * slopemap / (slopemap + slope_0)
#    pcr.report(rogrid1, os.path.join(wdir,'output','Preprocess_maps','rogrid1.map'))
    rogrid2 = runoff_c0 + rogrid1
#    pcr.report(rogrid2, os.path.join(wdir,'output','Preprocess_maps','rogrid2.map'))

    imp_map = ui.imp_map
    if imp_map in ('Y', 'y', 'yes', 'Yes', 'YES'):
        imp_map = pcr.readmap(os.path.join(wdir,'input','maps','imperviousness_start.map'))
        
        # Adapt RUNOFF using the IMPERVIOUS FRACTION (range 0-1) input
        rocoeff = rogrid2 * (pcr.scalar(mask) - imp_map) + imp_map
        
    else:
        ## LANDUSE DERIVED IMPERVIOUSNESS
        # Set IMPERVIOUS FRACTION (%) for urban cells
        imp_value = ui.imp_value
        # Adapt RUNOFF in urban cells using the IMPERVIOUS FRACTION (range 0-1) input
        urbanro = rogrid2 * (1 - imp_value) + imp_value
#        pcr.report(urbanro, os.path.join(wdir,'output','Preprocess_maps','urbanro.map'))
        # Calculate final runoff coefficients
        rocoeff = pcr.ifthenelse(pcr.nominal(landuse) == pcr.nominal(13), urbanro, rogrid2)

    pcr.report(rocoeff, os.path.join(wdir,'output','Preprocess_maps','runoff_co.map'))
    return rocoeff

def depression(slopemap, soil, landuse, mask, wdir):

#    print('')
#    print('')
#    print('------ DEPRESSION STORAGE CAPACITY calculation ------')
        
    # with spatially distributed imperviousness

    # Combine LANDUSE map (17 classes) with the SOIL map (12 classes) to calculate DEPRESSION map
    depression0 = pcr.lookupscalar(os.path.join(wdir,'input','tables','depression.tbl'), landuse, soil)
    
    b = ui.b
    Sdu = ui.Sdu
    imp_value = ui.imp_value    
    
    tempgrid = slopemap * pcr.scalar(-b)
    depressgrid = depression0 * pcr.exp(tempgrid)
    
    imp = ui.imp_map            
    if imp in ('Y', 'y', 'yes', 'Yes', 'YES'):
        imp_map = pcr.readmap(os.path.join(wdir,'input','maps','imperviousness.map'))
        depression = Sdu * imp_map + (pcr.scalar(mask) - imp_map) * depressgrid
    else:
        urbandepr = Sdu * imp_value + (1 - imp_value) * depressgrid
        # Calculate final DEPRESSION STORAGE CAPACITY
        depression = pcr.ifthenelse(pcr.nominal(landuse) == pcr.nominal(13), urbandepr, depressgrid)

    pcr.report(depression, os.path.join(wdir,'output','Preprocess_maps','depression.map'))

def velocity_func(manning_coeff, slopemap, radius, conv_factor, wdir):
    # default values in m/s
    defaultMinVelocity = 0.001
    defaultMaxVelocity = 3.0

    if ui.defaultVelocity in ("N","n"):
        minVelocity = ui.minVelocity
        maxVelocity = ui.maxVelocity

    else:
        minVelocity = defaultMinVelocity
        maxVelocity = defaultMaxVelocity
    
    # calculate velocity:
    # velocity = (1 / Manning) * Radius**(2/3) * (Slope)**(1/2)
    velocityTemp1 = (manning_coeff**(-1))*(radius**(2/3.))*pcr.sqrt(slopemap)
    
    # take into account the minimum velocity
    velocityTemp2 = pcr.ifthenelse(velocityTemp1 < minVelocity, pcr.scalar(minVelocity), velocityTemp1)
    # take into account the maximum velocity
    velocity = pcr.ifthenelse(velocityTemp2 > maxVelocity, pcr.scalar(maxVelocity), velocityTemp2)
    pcr.report(velocity, os.path.join(wdir,'output','Preprocess_maps','velocity_MperS.map'))
    velocity = velocity*conv_factor      # conv_factor is selected by the user (temporary way to have time step less than 1 hour)
    pcr.report(velocity,os.path.join(wdir,'output','Preprocess_maps','velocity.map'))
    return velocity

def t0_h(velocity, flowdir, flowacc, mask, conv_factor, wdir):

#    print('')
#    print('')
#    print('------ MEAN TIME TRAVEL TO CATCHMENT OUTLET (T0_h) calculation ------')
    
    # mask0 is a map that indicates the catchment outlet
    mask0 = pcr.boolean(pcr.ifthenelse(flowacc==pcr.mapmaximum(flowacc),mask,0))
    pcr.report(mask0, os.path.join(wdir,'output','Preprocess_maps','mask0.map'))
    friction = pcr.pcrpow(velocity*5./3., (-1))
    traveltime = pcr.ldddist(flowdir, mask0, friction)
    pcr.report(traveltime, os.path.join(wdir,'output','Preprocess_maps','t0_h.map'))
    # time travel in hours
    traveltimeH = traveltime*conv_factor/3600
    pcr.report(traveltimeH, os.path.join(wdir,'output','Preprocess_maps','t0_hH.map'))
    return traveltime

def delta_h(flowdir, velocity, slopemap, radius, wdir):

#    print('')
#    print('')
#    print('------ STANDARD DEVIATION of T0_h calculation ------')

    mask0 = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','mask0.map'))
    disp_coeff = velocity*radius/(slopemap*2.)
    pcr.report(disp_coeff, os.path.join(wdir,'output','Preprocess_maps','disp_coef.map'))
    celerity = velocity*5./3.
    pcr.report(celerity, os.path.join(wdir,'output','Preprocess_maps','celerity.map'))
    temp = disp_coeff*2./(pcr.pcrpow(celerity, 3))
    pcr.report(temp, os.path.join(wdir,'output','Preprocess_maps','temp.map'))
    # convolution
    delta_h_sqr = pcr.ldddist(flowdir, mask0, temp)
    pcr.report(delta_h_sqr, os.path.join(wdir,'output','Preprocess_maps','delta_h_sqr.map'))
    deltah = pcr.sqrt(delta_h_sqr)
    pcr.report(deltah, os.path.join(wdir,'output','Preprocess_maps','delta_h.map'))
    return deltah

def t0_s(flowdir, velocity, links, wdir):
#    print('')
#    print('')
#    print('------ MEAN TRAVEL TIME TO SUBCATCHEMNT OUTLET (T0_s) calculation ------')

    mask0 = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','mask0.map'))
    mainriver = pcr.scalar(pcr.operators.pcrNot(pcr.defined(links)))
    celerity = pcr.pcrpow(5./3.*velocity, (-1))*mainriver
    traveltime = pcr.ldddist(flowdir, mask0, celerity)
    pcr.report(traveltime, os.path.join(wdir,'output','Preprocess_maps','t0_s.map'))

def delta_s(flowdir, velocity, links, slopemap, radius, wdir):

#    print('')
#    print('')
#    print('------ STANDARD DEVIATION of T0_s calculation ------')

    mask0 = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','mask0.map'))
    mainriver = pcr.scalar(pcr.operators.pcrNot(pcr.defined(links)))
    disp_coeff = velocity*radius/(slopemap*2.)
    celerity = velocity*5./3.
    temp = disp_coeff*2./(pcr.pcrpow(celerity, 3))*mainriver
    # Convolution
    delta_s_sqr = pcr.ldddist(flowdir, mask0, temp)
    deltas = pcr.sqrt(delta_s_sqr)
    pcr.report(deltas, os.path.join(wdir,'output','Preprocess_maps','delta_s.map'))

def dynamic_preproc(t, dynamic_inputs):
    
    pcr.setglobaloption("double")
    
    wdir=dynamic_inputs['wdir']
    #wdir = r'E:\Master\Master_thesis\WetSpa_irrigated'
   
    answer = ui.timestep
    answer = answer.split(' ')
    # conversion factor (from seconds to whatever unit of measure is selected)
    number = float(answer[0])
    if answer[1] in ('y', 'Y'):
        conv_factor = number*365*24*3600
    elif answer[1] in ('m', 'M'):
        conv_factor = number*30*24*3600
    elif answer[1] in ('d', 'D'):
        conv_factor = number*24*3600
    elif answer[1] in ('h', 'H'):
        conv_factor = number*3600
    elif answer[1] in ('min', 'MIN', 'Min'):
        conv_factor = 60*number
    elif answer[1] in ('s', 'S', 'sec'):
        conv_factor = number
    else:
        print('you did not write the time step properly, I''m going to keep it hourly')
    
    
    pcr.setclone(os.path.join(wdir,'input\maps\CloneScalar1.map'))
    #pcr.setclone(r'E:\Master\Master_thesis\WetSpa_irrigated\input\maps\CloneScalar1.map')
    
    lu_fh=(os.path.join(wdir,'input\maps\lcc_WetSpa_%d.tif'))%(t)
    #lu_fh = r'E:\Master\Master_thesis\WetSpa_irrigated\input\maps\lcc_WetSpa_%d.tif' %(t)
    #dtype = "Int32"
    
    LU_pcr = lu_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = lu_fh,dst_filename = LU_pcr,ot = gdal.gdalconst.GDT_Int32,VS = "VS_NOMINAL")
    #LU_pcr = convert_tif_to_pcr(lu_fh, dtype)    
    
    
    mask = pcr.boolean(pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','mask.map')))        
    streamOrder = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','streamorder.map'))
    slopemap = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps', 'WGS84','slope_32643.map'))
    soil=pcr.readmap(dynamic_inputs['soil'])
    #soil = pcr.readmap(os.path.join(wdir,'input','maps','K3_soil_final.map'))
    total_mask = pcr.ifthenelse((mask==1), pcr.boolean(True), pcr.boolean(False))
    radius = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','radius.map'))
    flowacc = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','flowacc.map'))
    flowdir = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','flowdir.map'))
    links = pcr.readmap(os.path.join(wdir,'output','Preprocess_maps','link.map'))
    
  
    rootdepth = reclass_based_on_one_map(os.path.join(wdir,'input','tables','rootdepth.tbl'),LU_pcr)
    
    interc_min = reclass_based_on_one_map(os.path.join(wdir,'input','tables','interc_min.tbl'),LU_pcr)
    interc_max = reclass_based_on_one_map(os.path.join(wdir,'input','tables','interc_max.tbl'),LU_pcr)
    
    manning_coeff = manning(mask, streamOrder, LU_pcr, wdir)
    
    runoff_co = runoff_coeff(slopemap, soil, LU_pcr, mask, total_mask, wdir)
    
    depression(slopemap, soil, LU_pcr, mask, wdir)
    
    velocity = velocity_func(manning_coeff, slopemap, radius, conv_factor, wdir)
    
    t0_h_map = t0_h(velocity, flowdir, flowacc, mask, conv_factor, wdir)
    
    delta_h_map = delta_h(flowdir, velocity, slopemap, radius, wdir)
    
    t0_s(flowdir, velocity, links, wdir)
    
    delta_s(flowdir, velocity, links, slopemap, radius, wdir)
    
    fhs= glob.glob(os.path.join(wdir, 'output\Preprocess_maps','*.map'))
    #fhs = glob.glob(os.path.join(r'E:\Master\Master_thesis\WetSpa_irrigated\output\Preprocess_maps','*.map'))
    
    EPSG = '4326'
    for fh in fhs:
        convert_pcr_to_tif(fh, EPSG)