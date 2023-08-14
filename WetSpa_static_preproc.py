# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:56:46 2023

@author: esa
"""
import pcraster as pcr
import subprocess
import os
from osgeo import gdal, osr, ogr
import glob
import numpy as np


import UserInputs as ui


pcr.setglobaloption("lddin")

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

def project_map(map2project, EPSG):
    output_fh = map2project[:-4]+'_%s' %(EPSG) + map2project[-4:]
    gdal.Warp(output_fh,map2project,dstSRS='EPSG:%s' %(EPSG))
    return output_fh

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
    
def MatchProjResNDV(source_file, target_fhs, output_dir, resample = 'near', dtype = 'float32', scale = None, ndv_to_zero = False):
    '''
    Matches the projection, resolution and no-data-value of a list of target-files
    with a source-file and saves the new maps in output_dir.
    
    Parameters
    ----------
    source_file : str
        The file to match the projection, resolution and ndv with.
    target_fhs : list
        The files to be reprojected.
    output_dir : str
        Folder to store the output.
    resample : str, optional
        Resampling method to use, default is 'near' (nearest neighbour).
    dtype : str, optional
        Datatype of output, default is 'float32'.
    scale : int, optional
        Multiple all maps with this value, default is None.
    
    Returns
    -------
    output_files : ndarray 
        Filehandles of the created files.
    '''
    dst_info=gdal.Info(gdal.Open(source_file),format='json')
    output_files = np.array([])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for target_file in target_fhs:
        folder, fn = os.path.split(target_file)
        src_info=gdal.Info(gdal.Open(target_file),format='json')
        output_file = os.path.join(output_dir, fn)
        gdal.Warp(output_file,target_file,format='GTiff',
                      srcSRS=src_info['coordinateSystem']['wkt'],
                      dstSRS=dst_info['coordinateSystem']['wkt'],
                      srcNodata=src_info['bands'][0]['noDataValue'],
                      dstNodata=dst_info['bands'][0]['noDataValue'],
                      width=dst_info['size'][0],
                      height=dst_info['size'][1],
                      outputBounds=(dst_info['cornerCoordinates']['lowerLeft'][0],
                                    dst_info['cornerCoordinates']['lowerLeft'][1],
                                    dst_info['cornerCoordinates']['upperRight'][0],
                                    dst_info['cornerCoordinates']['upperRight'][1]),
                      outputBoundsSRS=dst_info['coordinateSystem']['wkt'],
                      resampleAlg=resample)
        output_files = np.append(output_files, output_file)
        if not np.any([scale == 1.0, scale == None, scale == 1]):
            driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(output_file)
            DATA = OpenAsArray(output_file, nan_values = True) * scale
            CreateGeoTiff(output_file, DATA, driver, NDV, xsize, ysize, GeoT, Projection)
        if ndv_to_zero:
            driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(output_file)
            DATA = OpenAsArray(output_file, nan_values = False)
            DATA[DATA == NDV] = 0.0
            CreateGeoTiff(output_file, DATA, driver, NDV, xsize, ysize, GeoT, Projection)
    return output_files    

def reclass_based_on_one_map(table, input_map):
    #it replaces the values in the map according to their corresponding values in the table
    parameter = pcr.lookupscalar(table,input_map)
    fname = table[0:-4].replace('tables','Preprocess_maps',1)
    fname = fname.replace('input', 'output',1)+'.map'
    pcr.report(parameter,fname)

def mask_func(elevation, wdir):
    #print('')
    #print('')
    #print('------ MASK CALCULATION ------')
    
    mask = pcr.ifthen(pcr.defined(elevation), pcr.boolean(1))
    pcr.report (mask, os.path.join(wdir,"output", "Preprocess_maps", "mask.map"))
    #mapconversion('mask', xul, yll, cellsize, ncols, nrows)
    return mask

def flow_dir(elevation, mask, wdir):
    class parameter:
        pass

    #initialization parameters
    Core_Depth = parameter()
    Core_Depth.name = 'Core_Depth'
    Core_Depth.value = 0
    Core_Volume = parameter()
    Core_Volume.name = 'Core_Volume'
    Core_Volume.value = 0
    Core_Area = parameter()
    Core_Area.name = 'Core_Area'
    Core_Area.value = 0
    Catchment_Precipitation = parameter()
    Catchment_Precipitation.name = 'Catchment_Precipitation'
    Catchment_Precipitation.value = 0
    
    
    def ask_parameter(param):
        question = 'value for '+ param.name + ' =  '
        while param.value == 0:
            try:
                param.value = int(input(question))
                break
            except:
                print('you should input an integer number, please try again')
        return param

    def ask_them_all():
        ask_parameter(Core_Depth)
        ask_parameter(Core_Volume)
        ask_parameter(Core_Area)
        ask_parameter(Catchment_Precipitation)

    def number_of_pit():
        global pits
        pit_map = pcr.uniqueid(pcr.ifthen(flowdir==5, mask))
        pit_number = pcr.mapmaximum(pit_map)
        pits = pcr.cellvalue(pit_number, 1, 1)[0]
        

    #print('')
    #print('')
    #print('------ FLOW DIRECTION CALCULATION ------')
    print('4 threshold values are need to be specified by the user for pits removing purposes')
    print('(for more information see PCRaster manual):')
    # Just for Oliver
    #ask_them_all()
    Core_Depth.value = ui.Core_Depth
    Core_Volume.value = ui.Core_Volume
    Core_Area.value = ui.Core_Area
    Catchment_Precipitation.value = ui.Catchment_Precipitation
    
    # calculation (first try)
    flowdir = pcr.lddcreate(elevation, Core_Depth.value, Core_Volume.value, Core_Area.value, Catchment_Precipitation.value)

   # check for multiple pits
    number_of_pit()

    while pits > 1:
        print('')
        print('with your parameter choice you obtained a flow direction map with \t', str(pits), ' pits')
        answer = input('Would you like to change your parameter values? (Y/N) ')
        if answer in ('Y', 'y'):
            Core_Depth.value = 0
            Core_Volume.value = 0
            Core_Area.value = 0
            Catchment_Precipitation.value = 0
            ask_them_all()
            flowdir = pcr.lddcreate(elevation, Core_Depth.value, Core_Volume.value, Core_Area.value, Catchment_Precipitation.value)
            number_of_pit()
        else:
            break
    # update user input
    filename = open(os.path.join(wdir, 'codes', "UserInputs.py"),"r")
    fileContent = filename.readlines()
    filename.close()
    count = 0
    for line in fileContent:
        if "Core_Depth" in line:
            fileContent[count] = "Core_Depth = %d\n" % (Core_Depth.value)
        if "Core_Volume" in line:
            fileContent[count] = "Core_Volume = %d\n" % (Core_Volume.value)
        if "Core_Area" in line:
            fileContent[count] = "Core_Area = %d\n" % (Core_Area.value)
        if "Catchment_Precipitation" in line:
            fileContent[count] = "Catchment_Precipitation = %d\n" % (Catchment_Precipitation.value)

        count +=1

    filename = open(os.path.join(wdir, 'codes', "UserInputs.py"),"w")

    for line in fileContent:
        filename.write(line)

    filename.close()
 
    pcr.report (flowdir, os.path.join(wdir,"output", "Preprocess_maps", "flowdir.map"))
    return flowdir

def flow_acc(flowdir,wdir):
    #print('')
    #print('')
    #print('------ FLOW ACCUMULATION calculation ------')
    
    #unit contribution of each cell:
    material = 1
    flowacc = pcr.accuflux(flowdir, material)
    pcr.report(flowacc, os.path.join(wdir,"output","Preprocess_maps","flowacc.map"))
    return flowacc

def stream_net(flowacc, wdir):
    #ask threshold
    #print('')
    #print('')
    #print('------ STREAM NETWORK calculation ------')

    cellthreshold = ui.Stream_threshold
    
    streamnet = pcr.ifthen(flowacc >= cellthreshold, pcr.boolean(1))
    pcr.report (streamnet,os.path.join(wdir,"output","Preprocess_maps","streamnet.map"))
    return streamnet

def stream_order(streamnet, flowdir, wdir):
    #print('')
    #print('')
    #print('------ STREAM ORDER calculation ------')
    
    # calculates the ldd of the streams
    lddstream = pcr.ifthen(streamnet, flowdir)
    # stream order calculation:
    streamOrder = pcr.streamorder(lddstream)
    pcr.report (streamOrder, os.path.join(wdir,"output","Preprocess_maps","streamorder.map"))
    return streamOrder

def slope_func(elev_WGS84, EPSG, wdir):
    #streamnet_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","streamnet.map"), '4326')
    streamnet_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","streamnet.map"), '4326')
    streamnet_projected_fh = project_map(streamnet_fh,EPSG)
    DEM_projected_fh = project_map(elev_WGS84,EPSG)
    
    Dem_pcr = DEM_projected_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = DEM_projected_fh,dst_filename = Dem_pcr,ot = gdal.gdalconst.GDT_Float32,VS = "VS_SCALAR")

    streamnet_pcr = streamnet_projected_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = streamnet_projected_fh,dst_filename = streamnet_pcr,ot = gdal.gdalconst.GDT_Int32,VS = "VS_NOMINAL")
    
    #dtype = "Float32"
    #elevation = convert_tif_to_pcr(DEM_projected_fh, dtype)
    #dtype = "Int32"
    #streamnet = convert_tif_to_pcr(streamnet_projected_fh, dtype)
    
    pcr.setclone(Dem_pcr)
    elevation = pcr.readmap(Dem_pcr)
    streamnet= pcr.readmap(streamnet_pcr)
    
    topostream = elevation*pcr.scalar(streamnet)
    landslope = pcr.slope(elevation)
    streamslope = pcr.slope(topostream)
    # combined slope map
    slopemap = pcr.cover(streamslope, landslope)
    
    # ask threshold
    min_slope = pcr.mapminimum(slopemap)
    #print('')
    #print('')
    #print('------ SLOPE calculation ------')
    print ('The minimum value of slope', pcr.cellvalue(min_slope, 1, 1)[0], 'is detected')
    answer = ui.Slope_min
    if answer in ('Y', 'y'):
        minslopethreshold = ui.Slope_threshold
        answer1 = ui.Slope_threshold_map
        if answer1 in ('Y', 'y'):
            slopearea_lower_threshold = pcr.ifthenelse (slopemap<minslopethreshold, pcr.boolean(1), pcr.boolean(0))
            pcr.report(slopearea_lower_threshold, os.path.join(wdir,"output","Preprocess_maps","slopethreshold_%s.map" %(EPSG)))
        slopemap = pcr.ifthenelse (slopemap < minslopethreshold, minslopethreshold, slopemap)

    pcr.report(slopemap, os.path.join(wdir,"output","Preprocess_maps","slope_%s.map" %(EPSG)))
    #map1_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","slopethreshold_%s.map" %(EPSG)), EPSG)
    #map2_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","slope_%s.map" %(EPSG)), EPSG)
    
    map1_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","slopethreshold_%s.map" %(EPSG)), EPSG)
    map2_fh = convert_pcr_to_tif(os.path.join(wdir,"output","Preprocess_maps","slope_%s.map" %(EPSG)), EPSG)
    
    output_files = MatchProjResNDV(elev_WGS84, [map1_fh, map2_fh], os.path.join(wdir, 'output', 'Preprocess_maps', 'WGS84'))
     
    return output_files[1]

def radius_func(flowacc, wdir):
    #ask for the storm return period
    #print('')
    #print('')
    #print('------ HYDRAULIC RADIUS calculation ------')
    return_period = ui.return_period
    #read the approapriate values
    if return_period in ('T2','t2'):
        radius_data = open(os.path.join(wdir,"input","tables","radius_T2.tbl"),"r")
        a = float(radius_data.readline())
        b = float(radius_data.readline())
    elif return_period in ('T10','t10'):
        radius_data = open(os.path.join(wdir,"input","tables","radius_T10.tbl"),"r")
        a = float(radius_data.readline())
        b = float(radius_data.readline())
    elif return_period in ('T100','t100'): 
        radius_data = open(os.path.join(wdir,"input","tables","radius_T100.tbl"),"r")
        a = float(radius_data.readline())
        b = float(radius_data.readline())
    elif return_period in ('Ttest', 'ttest'):
        radius_data = open(os.path.join(wdir,"input","tables","radius_Ttest.tbl"),"r")
        a = float(radius_data.readline())
        b = float(radius_data.readline())
    else:
        print('You did not type T2, T10 or T100!')
    
    #calculate hydraulic radius
    factor = 1000000      #conversion factor to have squared km
    radius = a * ((flowacc+1) * pcr.cellarea()/factor)**b
    pcr.report (radius, os.path.join(wdir,"output","Preprocess_maps","radius.map"))
    return radius

def streamlinks(flowdir,flowacc, wdir):

    #ask threshold to create new streamnet
    #print('')
    #print('')
    #print('------ STREAM LINKS calculation ------')

    cellthreshold = ui.Subcatchment_threshold
    newstreamnet = pcr.ifthenelse(flowacc<cellthreshold , pcr.boolean(0), pcr.boolean(1))
    # use function upstream: Sum of the cell values of its first upstream cell(s)
    upstreamMap = pcr.upstream(flowdir, pcr.scalar(newstreamnet))
    # Junctions arise when the upstream sum > 1
    junctions = pcr.ifthen(upstreamMap>1, pcr.boolean(1))
    # End points arise when the upstream sum = 0
    endpoints = pcr.ifthenelse(upstreamMap ==0, newstreamnet, 0)
    # apply uniqueid to make give every end point a unique value
    endpointsUnique = pcr.uniqueid(endpoints)
    # apply uniqueid to make give every junction a unique value
    junctionsUnique = pcr.uniqueid(junctions) 
    # map where junctions are converted to missing values and the rest of the map to 0
    junctionsMV= pcr.ifthen(pcr.defined(junctions) == False, pcr.scalar(0))
    # calculate stream parts starting from the ends
    resultMap = pcr.scalar(0)
    
    # apply for all end points
    for point in range(1,int(pcr.cellvalue(pcr.mapmaximum(endpointsUnique), 1, 1)[0]) + 1):
        # create map with one end point, the rest of the values 0
        endpoint = pcr.ifthenelse(endpointsUnique ==point, pcr.scalar(1), pcr.scalar(0))
        # create map with one end point and all junctions as missing values
        endpointmap = endpoint + junctionsMV
        # apply the path functiun starting at the end point and stopping at the first missing value it encounters
        pathMap = pcr.path(flowdir, pcr.boolean(endpointmap))
        # change the value of the path to a unique identifier
        pathMap = pcr.ifthenelse(pathMap, pcr.scalar(point), pcr.scalar(0))
        # add the new map to the result map, remove junction missing values
        resultMap = resultMap + pcr.ifthenelse(pcr.defined(pathMap), pathMap, pcr.scalar(0))

    # remove missing values from the map with unique values for the junctions
    junction = pcr.ifthenelse(pcr.defined(junctionsUnique), junctionsUnique, pcr.scalar(0)) 
    
    # apply for all junctions
    for i in range(1, int(pcr.cellvalue(pcr.mapmaximum(junctionsUnique), 1, 1)[0])+1):
        # create map with 1 junctions active, the rest of them are MV
        junctionMap = pcr.ifthenelse(junction == i, pcr.boolean(1), pcr.boolean(junctionsMV)) 
        # apply the path function starting at the junction and stopping at the first missing value it encounters
        pathMap = pcr.path(flowdir, junctionMap)
        # create a unique identifier
        number = pcr.mapmaximum(endpointsUnique) + i
        # change the values of the path to a unique identifier
        pathMap = pcr.ifthenelse(pathMap, pcr.scalar(number), pcr.scalar(0))
        # add the new map to the result map, remove junction missing values
        resultMap = resultMap + pcr.ifthenelse(pcr.defined(pathMap), pathMap, pcr.scalar(0))
        
    # change the 0 values to missing values
    links = pcr.ifthen(resultMap != 0, resultMap)
    pcr.report(links, os.path.join(wdir,"output","Preprocess_maps","link.map"))
    return links

def subwatershed(flowdir, links, wdir):
    #print('')
    #print('')
    #print('------ SUBWATERSHEDS calculation ------')
    watershedTemp = pcr.subcatchment(flowdir, pcr.ordinal(links))
    watershed = pcr.ifthen(watershedTemp != 0, watershedTemp)
    pcr.report(watershed, os.path.join(wdir,"output","Preprocess_maps","watershed.map"))
    return watershed

def initial_moisture(flowacc, slopemap, mask, wdir):
    # 1 added to each raster cell value to compute log of non-zero values
    #???? not completely sure but 1 is added in the ArcView version as well
    newflowacc = flowacc+1
    # Calculation of Topographical Wetness Index (TWI); Moore et al. (1993)
    factor = 100    # factor to have slope in %:
    TWI = pcr.ln(newflowacc*pcr.cellarea()/(slopemap*factor))
    TWI_min = pcr.mapminimum(TWI)
    TWI_max = pcr.mapmaximum(TWI)
                
    # Minimum relative saturation depending upon simulation and the study area, defined by the user
    #print('')
    #print('')
    #print('------ INITIAL SOIL MOISTURE calculation ------')
    Smin = ui.Smin
    # Assuming maximum relative moisture corresponding to full saturation 
    Smax = ui.Smax
    # Assumption of relating minimumm and maximum moisture content with TWI to get Initial Moisture Content
    #I have changed 0.8 in 0.85
    TWI_UpperLimit = TWI_max*0.85
    aa = (Smax-Smin)/(TWI_UpperLimit-TWI_min)
    bb1 = aa*TWI_min
    bb = Smin-bb1
    Initial_Moisture_temp = (TWI*aa)+bb
    Initial_Moisture = pcr.ifthenelse(TWI>TWI_UpperLimit, pcr.scalar(mask), Initial_Moisture_temp)
    
    pcr.report(Initial_Moisture, os.path.join(wdir,"output","Preprocess_maps","sm.map"))

###This is for running the static preprocessing jupyter notebook 
def static_preproc(static_inputs):
    
    #inputs from jupyter notebook 
    wdir=static_inputs['wdir']
    input_files = static_inputs['input_files']
    
    # convert DEM from tiff to PCRaster
    
    #wdir = r'E:\Master\Master_thesis\WetSpa-Python_v1_0'
    Dem_fh=input_files['DEM']
    #Dem_fh = r'E:\Master\Master_thesis\WetSpa-Python_v1_0\input\maps\K3_DEM_fill_sinks3.tif'
    Soil_fh=input_files['soil']
    #Soil_fh = r'E:\Master\Master_thesis\WetSpa-Python_v1_0\input\maps\K3_soil_new.tif'


    Dem_pcr = Dem_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = Dem_fh,dst_filename = Dem_pcr,ot = gdal.gdalconst.GDT_Float32,VS = "VS_SCALAR")

    soil = Soil_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = Soil_fh,dst_filename = soil,ot = gdal.gdalconst.GDT_Int32,VS = "VS_NOMINAL")

    # Set clonemap for all maps
    pcr.setclone(os.path.join(wdir,"input\maps\CloneScalar1.map"))
    #pcr.setclone(r'E:\Master\Master_thesis\WetSpa-Python_v1_0\input\maps\CloneScalar1.map')

    # create a mask
    elevation = pcr.scalar(pcr.readmap(Dem_pcr))
    mask_pcr = mask_func(elevation, wdir)

    # flow direction
    flowdir = flow_dir(elevation, mask_pcr, wdir)

    # flow accumulation
    flowacc = flow_acc(flowdir, wdir)

    # stream network
    streamnet = stream_net(flowacc, wdir)

    # stream order
    streamOrder= stream_order(streamnet, flowdir, wdir)

    # slope
    EPSG = '32643'
    slope_fh = slope_func(Dem_fh, EPSG, wdir)
    
    pcr.setclone(os.path.join(wdir,"input\maps\CloneScalar1.map"))
    #pcr.setclone(r'E:\Master\Master_thesis\WetSpa-Python_v1_0\input\maps\CloneScalar1.map')
    # radius
    radius = radius_func(flowacc, wdir)

    # stream links
    links = streamlinks(flowdir, flowacc, wdir)

    # watershed
    watershed = subwatershed(flowdir, links, wdir)

    # conductivity derived from soil types
    #print('')
    #print('')
    #print('------ CONDUCTIVITY calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','conductivity.tbl'), soil)

    # porosity derived from soil types
    #print('')
    #print('')
    #print('------ POROSITY calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','porosity.tbl'), soil)

    # field capacity derived from soil types
    #print('')
    #print('')
    #print('------ FIELD CAPACITY calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','fieldcap.tbl'), soil)

    # residual moisture derived from soil types
    #print('')
    #print('')
    #print('------ RESIDUAL MOISTURE calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','residual.tbl'),soil)

    # pore index derived from soil types
    #print('')
   #print('')
    #print('------ PORE DISTRIBUTION INDEX calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','poreindex.tbl'),soil)

    # wilting point derived from soil types
    #print('')
    #print('')
    #print('------ WILTING POINT calculation ------')
    reclass_based_on_one_map(os.path.join(wdir,'input','tables','wilting.tbl'),soil)

    # initial
    slope_pcr = slope_fh[:-4]+'.map'
    ConvertToPCRaster(src_filename = slope_fh,dst_filename = slope_pcr,ot = gdal.gdalconst.GDT_Float32,VS = "VS_SCALAR")
    #slope_pcr = convert_tif_to_pcr(slope_fh, "Float32")
    slopemap = pcr.readmap(slope_pcr)
    initial_moisture(flowacc, slopemap, mask_pcr, wdir)

    EPSG = '4326'
    fhs = glob.glob(os.path.join(wdir, 'output\Preprocess_maps','*.map'))
    #fhs = glob.glob(os.path.join(r'E:\Master\Master_thesis\WetSpa-Python_v1_0\output\Preprocess_maps','*.map'))
    for fh in fhs:
        convert_pcr_to_tif(fh, EPSG)