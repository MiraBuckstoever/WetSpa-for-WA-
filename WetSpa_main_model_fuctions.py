# import packages

import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import warnings
import netCDF4
import dask.array as da 
from osgeo import gdal, osr, ogr
import math
from PIL import Image

from WetSpa_dynamic_preproc import dynamic_preproc

import global_parameters as gp

from dask.distributed import Client, LocalCluster
import multiprocessing as mp

# functions 
def start_multiprocessing():
    try:
        client = Client('tcp://localhost:8786', timeout='4s')
        return client
    except OSError:
        cluster =  LocalCluster(ip="",n_workers=int(0.9 * mp.cpu_count()),
            scheduler_port=8786,
            processes=False,
            threads_per_worker=4,
            memory_limit='48GB',
        )
    return Client(cluster)


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
    

def open_nc(nc,chunks):
    with xr.open_dataset(nc, chunks=chunks) as dts:
        key=list(dts.keys())[0]
        var=dts[key]
        dts.close()
        return var,key

def open_nc_not(nc,chunks):
    lonlat = ['longitude','latitude']
    lonlat_chunks = {k: v for k, v in chunks.items() if k in lonlat}
    with xr.open_dataset(nc, chunks=lonlat_chunks) as dts:
        key=list(dts.keys())[0]
        var=dts[key]
        dts.close()
        return var,key
    

def area_sqkm(ds):
    '''
    Calculate pixel area in square kilometers
    '''
    dlat = abs(ds.latitude[-1]-ds.latitude[-2])
    dlon = abs(ds.longitude[-1]-ds.longitude[-2])

    R = 6378137 # Radius of earth  6.37e6 -> this is in m 

    # we know already that the spacing of the points is one degree latitude
    dϕ = np.deg2rad(dlat)
    dλ = np.deg2rad(dlon)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(ds.latitude))
    # pixel area in square meter
    pixel_area = dA.where(ds.notnull())
    # pixel area in square kilometer
    pixel_area = pixel_area/1e6 ## Area in square kilometers
    ds.close()
    return pixel_area

#function to compute Nash-Sutcliffe Efficiency
def nash_sutcliffe(observed, simulated):
    x = observed
    y = simulated
    if(len(x)!=len(y)):
        print('observed and simulated serieses are not equal in length')
    part1 = ((y-x)**2).sum()
    part2 = ((x - x.mean())**2).sum()
    ns = 1. - part1 / part2
    return ns

def kge(evaluation, simulation, return_all=False):
    """
    Kling-Gupta Efficiency
    Corresponding paper: 
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: 
        Implications for improving hydrological modelling
    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation 
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    if len(evaluation) == len(simulation):
        cc = np.corrcoef(evaluation, simulation)[0, 1]
        alpha = np.std(simulation) / np.std(evaluation)
        beta = np.sum(simulation) / np.sum(evaluation)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        print("evaluation and simulation lists do not have the same length.")
        print('evaluation :',len(evaluation), 'simulation :',len(simulation))
        return np.nan
 
def nc2GeoTiff(dataset, template_fh):
    latchunk = 200
    lonchunk = 200
    timechunk = 1
    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk} 
    data_nc, _ = open_nc(dataset, chunks)
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(template_fh)
    for t in range(len(data_nc['time'])):
        data = data_nc.isel(time=t).values
        data = np.flipud(data)
        #f_name = dataset[:-3]+'_'+str(data_nc.time[t].values)[:10]+'.tif'
        f_name = dataset[:-3]+'_'+str(t)+'.tif'
        CreateGeoTiff(f_name,data, driver, NDV, xsize, ysize, GeoT, Projection)
        
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

def _log_create(file): #taken from PixSWAB
        time_now = datetime.datetime.now()
        time_str = time_now.strftime('%Y-%m-%d %H:%M:%S.%f')
        txt = '{t}: WetSpa model'.format(t=time_str)
        fp = open(file, 'w+')
        fp.write('{}\n'.format(txt))
        return fp

def write(fp, msg=''): #taken from PixSWAB
        txt = '{msg}'.format(msg=msg)
        fp.write('{}\n'.format(txt))
        
def _log_close(fp): #taken from PixSWAB
        time_now = datetime.datetime.now()
        time_str = time_now.strftime('%Y-%m-%d %H:%M:%S.%f')
        txt = '{t}: WetSpa model finished.'.format(t=time_str)
        fp.write('{}\n'.format(txt))
        fp.close()
    
## Lambda functions
sum2 = lambda a,b: a+b
dif2 = lambda a,b: a-b
quo2 = lambda a,b: a/b
mul2 = lambda a,b: a*b
perc2 = lambda a,b: 100/a*b
##---------------##
## WetSpa model ##
##---------------##


#%%
def wetspa_model(wetspa_inputs):
    print(5)
    '''
    
    '''
    ## Switch off the warnings
    warnings.filterwarnings("ignore", message='invalid value encountered in greater')
    warnings.filterwarnings("ignore", message='divide by zero encountered in true_divide')
    warnings.filterwarnings("ignore", message='invalid value encountered in true_divide')
    warnings.filterwarnings("ignore", message='overflow encountered in exp')
    warnings.filterwarnings("ignore", message='overflow encountered in power')
    warnings.filterwarnings("ignore", message='invalid value encountered in subtract')
    warnings.filterwarnings("ignore", message='All-NaN slice encountered')
    warnings.filterwarnings("ignore", message='xarray.ufuncs is deprecated')
    warnings.filterwarnings("ignore", message='invalid value encountered in power')

#    wdir = r'E:\Master\Master_thesis\WetSpa-Python_v1_0'    
    wdir=wetspa_inputs['wdir']
    period = wetspa_inputs['period']
    
#    Startdate='2010-06-01'
#    Enddate='2017-05-31'

#    period = {
#             's': Startdate,
#             'e': Enddate
#             }
    
    y = datetime.datetime.strptime(period['s'], '%Y-%m-%d').year # starting year of the simulation
#    latchunk = 1000
#    lonchunk = 1000
#    timechunk = 1
#    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk} 

    # Get list of input files and chunks size 
    input_files = wetspa_inputs['input_files']
    chunks = wetspa_inputs['chunks']

    # read the netCDF files
    Pt,_= open_nc(input_files['p_in'],chunks)
    LU,_= open_nc(input_files['lu_in'],chunks)
    ETP,_ = open_nc(input_files['pet_in'],chunks)
    ET_ac,_=open_nc(input_files['et_act_in'],chunks)   
    Tmi,_ = open_nc(input_files['tmin_in'],chunks)
    Tma,_ = open_nc(input_files['tmax_in'],chunks)
    
    template_fh = input_files['DEM']
    LU_nc=input_files['lu_in']
    
    
    # create GeoTiffs of the landuse 
    nc2GeoTiff(LU_nc, template_fh)
    
    
    # extract the dataset for the start and end time
    Pt = Pt.sel(time=slice(period['s'],period['e']))
    ETP = ETP.sel(time=slice(period['s'],period['e']))
    ET_ac = ET_ac.sel(time=slice(period['s'], period['e']))
    Tmi = Tmi.sel(time=slice(period['s'], period['e']))
    Tma = Tma.sel(time=slice(period['s'], period['e']))
    
    ## check alignments of the datasets
    try:
        xr.align(Pt,ETP, join='exact')   # will raise a ValueError if not aligned
    except:
        print('the {0}, and {1} dataset do not align.'.format('precipitation', 'potential evapotranspiration'))

        
    if((LU.time[0]<=np.datetime64(period['s'])) & (LU.time[-1]>=np.datetime64(period['e']))):
        LU = LU.sel(time=slice(period['s'],period['e']))
    
    ## check alignments of the the lcc dataset
    try:
        xr.align(LU, Pt[0], join='exact')   # will raise a ValueError if not aligned
    except:
        print('the {0} dataset does not align.'.format('land cover class'))

    # get variables needed for ET0 (Ra and kc for the crops )
    ET0_dict =wetspa_inputs['ET0_dict']
    
    # list of output requested
    output_list = wetspa_inputs['output']  
    
    # DO THE COMPUTATION

    # Initialize some variables
    zar = Pt[0]*0  # zero array with the dimension of the inputs
    ar1 = 1+zar  ## arrays of one
    tol = zar+0.999
    mintol = 0.000001
    
    F = v_ei = v_interc = s_int = v_netp = v_run = v_rs = v_infil = v_rs1 = s_dep = v_ed = v_depre = v_rn1 = v_ri = v_es = v_perco = cp = et = v_eg_spa = etg = etb= zar
    et_irr= et_n0 = et_nirr = et_nirr_n0 = ET_act_n0 = et_perc = et_perc_nirr = zar
    
    parameters = wetspa_inputs['parameters']
    # calibration parameters
    Kep = parameters['k_ep']
    k_run = parameters['k_run'] # coefficent for low rainfall intensity
    P_max = parameters['P_max'] # threshold of rainfall intensity
    k_ss = parameters['k_ss']# initial soil moisture coefficient
    ki = parameters['ki']# interflow scaling factor
    dt = parameters['dt_hours'] # time step in hours
    g0 = parameters['g0']
    g_max = parameters['g_max']
    kg = parameters['kg']
    kcw=parameters['kcw']
    kcf=parameters['kcf']
    ieff=parameters['ieff']
                     
    # Select years from monthly time index
    tt=pd.to_datetime(Pt.time)
    ss= tt.year
    seen = set()
    seen_add = seen.add
    years = [x for x in ss if not (x in seen or seen_add(x))]
    

    # parameter maps
    #tif files are flipped when read in -> not sure why -> this needs to be fixed -> PC Raster might be the reason 
    mask = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','mask.tif'), nan_values = -3.4028235e+38)
    mask =np.flipud(mask)
    porosity = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','porosity.tif'), nan_values = -3.4028235e+38)
    porosity =np.flipud(porosity)
    sm_init = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','sm.tif'), nan_values = -3.4028235e+38)
    sm_init =np.flipud(sm_init)
    fieldcap = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','fieldcap.tif'), nan_values = -3.4028235e+38)
    fieldcap =np.flipud(fieldcap)
    slope = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','WGS84', 'slope_32643.tif'), nan_values = -3.4028235e+38)
    slope =np.flipud(slope)
    conductivity = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','conductivity.tif'), nan_values = -3.4028235e+38)
    conductivity =np.flipud(conductivity)
    residual = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','residual.tif'), nan_values = -3.4028235e+38)
    residual =np.flipud(residual)
    poreindex = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','poreindex.tif'), nan_values = -3.4028235e+38)
    poreindex =np.flipud(poreindex)
    wilting = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','wilting.tif'), nan_values = -3.4028235e+38)
    wilting =np.flipud(wilting)
    watershed = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','watershed.tif'), nan_values = -3.4028235e+38)
    watershed =np.flipud(watershed)
    links = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','link.tif'), nan_values = -3.4028235e+38)
    links =np.flipud(links)
    
    Pixel_areas_km2 = area_sqkm(Pt[0])
    area = np.nanmean(Pixel_areas_km2.values)*1000000 # cell surface in m2
    nsub = int(np.nanmax(watershed))
    lenp = len(Pt.time)
    
    qs = np.zeros(lenp)
    et_sup = np.zeros(lenp)
    qi = np.zeros(lenp)
    qg = np.zeros(lenp)
    rg_h = np.zeros(lenp)
    gt_h = np.zeros(lenp)
    eg_h = np.zeros(lenp)
    et_h = np.zeros(lenp)
    
    qg_s = np.zeros((lenp,nsub))
    v_rg = np.zeros((lenp,nsub))
    v_eg = np.zeros((lenp,nsub))
    gt_s =np.zeros((lenp,nsub))
    pet_s = np.zeros((lenp,nsub))
    perco_s =np.zeros((lenp,nsub))
    et_s = np.zeros((lenp,nsub))
    gt_s[0,:] = g0
    
    kg_sub = np.zeros(nsub)
    slope_sub = np.zeros(nsub)

    ncell = np.nansum(mask.astype('int'))
    
    #spatial distribution entries needed for spatial distribution of v_eg
    watershed_xr=xr.DataArray(watershed)
    z=[a for a in (np.unique(watershed_xr.values)) if str(a) !='nan']
    
    
    for i in range(nsub):
        slope_sub[i] = np.nanmean(slope[watershed == i+1]) # subcatchment slope
        kg_sub[i] = kg*np.nanmean(slope) # base flow recession coefficient
  
    for j in range(len(years)): 
        
        Soil_fh=input_files['soil']
        soil = Soil_fh[:-4]+'.map'
        
        dynamic_paras={
            'wdir':wdir, 
            'soil':soil}
        
        dynamic_preproc(j, dynamic_paras)  

        # prameter maps
        #tif files are fliped when read in -> not sure why -> this should be fixed 
        interc_min = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','interc_min.tif'), nan_values = -3.4028235e+38)
        interc_min=np.flipud(interc_min)
        interc_max = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','interc_max.tif'), nan_values = -3.4028235e+38)
        interc_max=np.flipud(interc_max)
        rootdepth = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','rootdepth.tif'), nan_values = -3.4028235e+38)
        rootdepth=np.flipud(rootdepth)
        runoff_co = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','runoff_co.tif'), nan_values = -3.4028235e+38)
        runoff_co=np.flipud(runoff_co)
        depression = OpenAsArray(os.path.join(wdir,'output','Preprocess_maps','depression.tif'), nan_values = -3.4028235e+38)   
        depression = np.flipud(depression)
        
        porosity_volume = porosity*rootdepth*1000.  
        residual_volume = residual*rootdepth*1000.
        porosity_volume = porosity*rootdepth*1000.
        fieldcap_volume = fieldcap*rootdepth*1000.
        wilting_volume = wilting*rootdepth*1000.
   
 
        if (j==0):
            mn_len_bf = 0
            months=Pt.time.where(Pt.time.dt.year.isin(years[j]),drop=True)
            
            if k_ss<0.:
                soil_moisture = sm_init*porosity_volume
      
            else:
                fieldcap_volume = fieldcap*rootdepth*1000.
                soil_moisture = fieldcap_volume*k_ss #k_ss is soil moisture coefficient
                
        else:
            mn_len_bf = len(Pt.time.where(Pt.time.dt.year.isin(years[:j]),drop=True))
            months=Pt.time.where(Pt.time.dt.year.isin(years[j]),drop=True)

        if(len(LU.time)==0):
           lu = LU.isel(time=0)
        elif(j < len(LU.time)):
           lu = LU.isel(time=j)
        else:
           lu = LU.isel(time = len(LU.time)-1)
        
        #masks for land uses that need to be adjusted 
        #mask for waterbodies 
        mask_wb = np.where((lu==17), 1, 0)
        
        #mask for irrigated forest 
        mask_if = np.where((lu==18), 1, 0)
        
        #mask for irrigated crop 
        mask_ic = np.where((lu==19), 1, 0)

        
        for i in range(len(months)):
            mon=(datetime.datetime.strptime((np.datetime_as_string((months.values[i]), unit='D')), '%Y-%m-%d')).month
            #print(mon)
            t= i+mn_len_bf
            print('\rmonth: {0} of {1}'.format(t+1, len(Pt.time)), end='')
            SRO = ET = ET_diff = ETb = ETg = Sup = ET_diff_irr = zar
            ETi= ETs= ETd = ETgw = ETirr = SM = ET_per = ET_per_irr = zar 
  
            P = Pt.isel(time=t)
            ETp = ETP.isel(time=t)*Kep
            ET_act = ET_ac.isel(time=t)
            Tmin = Tmi.isel(time=t)
            Tmax = Tma.isel(time=t)
            
            #Calculation of reference Evapotranspiration (ET0) based on  Hargreaves equation
            ET0=(0.0023*0.408*ET0_dict[mon][1])*(((Tmax.values+Tmin.values)/2)+17.8)*np.sqrt((Tmax.values-Tmin.values))
            
            #capacity=int_min=int_max=zar
            # interception computation
            nd = pd.Timestamp(P.time.values).timetuple().tm_yday #day of the year
            #capacity = interc_min + (interc_max - interc_min)*pow((0.5 + 0.5*math.sin(2.0*np.pi*(nd - 87.0)/365.0)),1.35)
            capacity = interc_min + (interc_max - interc_min)*math.pow((0.5 + 0.5*math.sin(2.0*np.pi*(nd - 87.0)/365.0)),1.35)

            condition1 = P>0.0
            applicable_area = np.logical_or(condition1,s_int > 0.0)
            
            v_ei = np.where(applicable_area, \
                                 (np.where((ETp < s_int),np.copy(ETp),np.copy(s_int))), \
                                 0.0)
            
            v_interc = np.where(applicable_area, \
                                 (np.where((P <= (capacity - s_int)), \
                                           np.copy(P), \
                                           np.maximum(0.0,capacity - s_int))), \
                                 zar)
            
            s_int = np.where(applicable_area, np.maximum(0.0, s_int + v_interc - v_ei),0.0)
            
            v_netp = np.where(condition1, P - v_interc, 0.0)    
            
            
            # Rain excess and infiltration computation
            condition = v_netp>0.0

            cp = np.where(condition, np.where(v_netp >= P_max, 1.0, \
                          k_run - (k_run-1.0)*P/P_max), cp)
            
    
            applicable_area = runoff_co>0.99
            v_run = np.where(condition, np.where(applicable_area, 1.0,\
                                      runoff_co*pow(soil_moisture/porosity_volume,cp)), v_run)

            # Surface runoff calculation
            v_rs = np.where(condition, v_netp*v_run, 0)

            # Infiltration calculation
            v_infil = np.where(condition, v_netp*(1.0-v_run), 0)

            # soil moisture update
            soil_moisture = np.where(condition, soil_moisture + v_infil, soil_moisture)

            # check for saturated condition
            v_rs = np.where(condition, np.where(soil_moisture>porosity_volume, v_rs+soil_moisture-porosity_volume, np.copy(v_rs)), 0)
            v_infil = np.where(condition, np.where(soil_moisture>porosity_volume, v_infil-soil_moisture+porosity_volume, np.copy(v_infil)), 0)
            soil_moisture = np.where(condition, np.where(soil_moisture>porosity_volume, np.copy(porosity_volume), soil_moisture), soil_moisture)
            
            # evaporation from depression storage
            v_ed = np.where(np.logical_and(v_rs>0.0, depression>0.0),  np.where((ETp-v_ei)<s_dep, ETp-v_ei, \
                                                                                      np.copy(s_dep)), \
                                     0.0)
            s_dep1 = np.copy(s_dep)
            v_crain =  v_rs-depression*np.log(1.-s_dep/depression)
            #to make sure that invalid values are not encountered 
            #v_crain = v_rs - depression * np.where((s_dep/depression >= 1) | (depression==0), np.log(1e-10), np.log(1. - s_dep / depression))
            #print('v_crain', np.unique(v_crain))

            # depression flux calculation
            v_depre = np.where(np.logical_and(v_rs>0.0, depression>0.0), v_rs*np.exp(0.0 - v_crain/depression), 0.0)
    
            # depression storage calculation
            s_dep= np.where(np.logical_and(v_rs>0.0, depression>0.0), np.maximum(0.0, s_dep+v_depre-v_ed), s_dep)
    
            #depression flux correction if depression storage>max depression storage capacity
            v_depre = np.where(np.logical_and(v_rs>0.0, depression>0.0), np.where(s_dep >=depression, depression-s_dep1, v_depre),0.0)
    
            # depression storage correction if depression storage>max depression storage capacity
            s_dep= np.where(np.logical_and(v_rs>0.0, depression>0.0), np.where(s_dep >=depression, np.copy(depression), s_dep), s_dep)
    
            #runoff is decreasing because we filled depressions
            v_rs1 = np.where(np.logical_and(v_rs>0.0, depression>0.0), np.maximum(0.0,v_rs - v_depre), np.copy(v_rs))
    
            # evaporation from depression storage
            v_ed = np.where(np.logical_and(v_rs>0.0, depression>0.0), v_ed, \
                                 np.where(s_dep>0.0,np.where((ETp-v_ei)<s_dep, ETp-v_ei, \
                                                                                      np.copy(s_dep)), \
                                          0.0))
            v_run1 =runoff_co*soil_moisture/porosity_volume
    
            # infiltration from depression storage
            # I have added a check for saturated conditions
            v_infil1 = np.where(np.logical_and(v_rs>0.0, depression>0.0), np.copy(v_infil), \
                                 np.where(np.logical_and(s_dep>0.0,soil_moisture<porosity),np.maximum(0.0, (s_dep-v_ed)*(1.0-v_run1)), np.copy(v_infil)))
            
            # depression storage is decreased because of evaporation
            s_dep = np.where(np.logical_and(v_rs>0.0, depression>0.0), s_dep, \
                                 np.where(s_dep>0.0,np.maximum(0.0, s_dep+v_depre-v_ed), 0.0))
    
            # soil moisture update
            soil_moisture = np.where(np.logical_and(v_rs>0.0, depression>0.0), np.copy(soil_moisture), \
                                 np.where(s_dep>0.0,soil_moisture+v_infil1, np.copy(soil_moisture)))

            # interflow calculation
            # I add a check to not allow v_ri to have negative values
            v_ri = np.where(soil_moisture>fieldcap_volume, \
                                     ki*slope*(conductivity*dt)*pow((soil_moisture-residual_volume)/(porosity_volume-residual_volume),poreindex), \
                                     0.0)

            # maximum possible and again not negative values
            v_ri = np.where(v_ri>(soil_moisture-residual_volume), soil_moisture-residual_volume, v_ri)
            v_ri = np.where(v_ri<0., 0., np.copy(v_ri))

            # soil moisture update
            soil_moisture = soil_moisture - v_ri     

            # evapo(transpi)ration calculation
            v_es = np.where(soil_moisture> residual_volume, \
                                     np.where(soil_moisture< fieldcap_volume, \
                                              (ETp-v_ei-v_ed)*((soil_moisture-wilting_volume)/(fieldcap_volume-wilting_volume)), \
                                              ETp-v_ei-v_ed), \
                                     v_es)

            # evaporation is decreased of 70% in urban areas
            # (when the maximum interception capacity is very very low)
            # it should be changed based on imperviousness
            v_es = np.where(interc_max < 0.01, v_es*0.3, np.copy(v_es))
    
            # maximum possible (residual soil moisture cannot evaporate)
            v_es = np.where(v_es>(soil_moisture-residual_volume),soil_moisture-residual_volume, np.copy(v_es))
    
            # not negative values
            v_es = np.where(v_es<0., 0.0, np.copy(v_es))
    
            # if the potential evapotranspiration is 0 --> there cannot be evaporation from the soil
            v_es = np.where(ETp==0.0, 0.0, np.copy(v_es))
    
            # soil moisture update
            soil_moisture = soil_moisture - v_es

            # percolation calculation
            v_perco = np.where(soil_moisture>residual_volume, \
                                        (conductivity*dt)*pow((soil_moisture-residual_volume)/(porosity_volume-residual_volume),poreindex),\
                                        0.0)
            
            # maximum possible and non-negative values
            v_perco = np.where(v_perco > (soil_moisture-residual_volume), soil_moisture-residual_volume, v_perco)
            v_perco = np.where(v_perco<0., 0.0, v_perco)
            
            # soil moisture update
            soil_moisture = soil_moisture - v_perco
            
            #et for water bodies (kc*ET0)
            et_wb=np.where((mask_wb==1), (kcw*ET0), 0)
            
            #Calculation of evapotranspiration
            et = v_ed + v_es + v_ei + et_wb
            
            et_h[t] = np.nanmean(et)
            # no routing 
            # computations should be checked
            #qs.append(np.nansum(v_rs*Pixel_areas_km2/1000)) # MCM
            qsx=np.nansum(v_rs*Pixel_areas_km2/1000) #MCM
            qs[i]=qsx
            #qi.append(np.nansum(v_ri*Pixel_areas_km2/1000)) # MCM  
            qix=np.nansum(v_ri*Pixel_areas_km2/1000) #MCM 
            qi[i]=qix

            
            # groundwater
            for n in range(nsub):
                pet_s[t, n] = np.nanmean(ETp.values[watershed == (n+1)])
                et_s[t, n] = np.nanmean(et[watershed == (n+1)])
                perco_s[t, n] = np.nanmean(v_perco[watershed == (n+1)])
                
                if t==0:
                    v_rg[t, n] = kg_sub[i]*g0
                    v_eg[t,n] = 0
                else:
                    # the groundwater flux per unit area is a function of the baseflow recession coefficient and of the groundwater storage of the previous time step
                    v_rg[t, n] = kg_sub[n]*gt_s[t-1,n]
                    
                    # evaporation from the saturated zone
                    v_eg[t, n] = (pet_s[t, n]-et_s[t, n])*gt_s[t-1, n]/g_max   # control for negative values

                # the groundwater flux per unit area is converted in [m3/s] (dt is in hours, area in m2)
                qg_s[t, n] = (v_rg[t, n]/1000.)*(np.nansum((mask[watershed == (n+1)].astype('int'))))*area/dt/3600.

                # control for negative values
                v_eg[t, n] = np.where(v_eg[t, n]<0, 0.0, v_eg[t, n])
                #print(np.unique(v_eg))
                #print(v_eg)
    
                # groundwater storage update
                if t==0:
                    gt_s[t, n] = g0 + perco_s[t, n] - v_rg[t, n] - v_eg[t, n]
                else:
                    gt_s[t, n] = gt_s[t-1,n] + perco_s[t, n] - v_rg[t, n] - v_eg[t, n]
    
                # catchment groundwater flow average
                rg_h[t] = rg_h[t] + v_rg[t, n]*(np.nansum(mask[watershed == (n+1)].astype('int')))/ncell
    
                # catchment groundwater storage average
                gt_h[t] = gt_h[t] + gt_s[t, n]*(np.nansum(mask[watershed == (n+1)].astype('int')))/ncell
    
                # catchment evapotranspiration avarages (update)
                eg_h[t] = eg_h[t]+v_eg[t, n]*(np.nansum(mask[watershed == (n+1)].astype('int')))/ncell
             
            #spatial distribution for v_eg   
            for n in z: #this is to make sure to exlude the nan values as they are staying
                #pixel=np.count_nonzero(watershed_xr==n)
                v_eg_spa=np.where(watershed_xr==n,\
                                  (v_eg[i,int(n-1)]/np.count_nonzero(watershed_xr==n)),\
                                      v_eg_spa)
            
            #recalculation of et (based on GW)
            et=et+v_eg_spa
                   
            #crop evapotranspiration for irrigated forest and irrigated crops (here: maize is used for all irrigated crops besides forest)
            #including irrigation efficiency 
            cet_if=np.where((mask_if==1), ((kcf*ET0)/ieff), 0)
            cet_ic=np.where((mask_ic==1), ((ET0_dict[mon][2]*ET0)/ieff), 0)
            
           #if crop evapotranspiration higher than calculated evapotranspiration: additional evapotranspiration from irrigation
            et_if=np.where(((mask_if==1)&(cet_if>et)), cet_if, 0)
            et_ic=np.where(((mask_ic==1)&(cet_ic>et)), cet_ic, 0)
           
            #Calculate supply for irrigated water bodies 
            sup_if=et_if/ieff
            sup_ic=et_ic/ieff
            
            sup=sup_if+sup_ic
            
            #recalculation of et (based on irrigation)
            et=et+et_if+et_ic
            
            #et without addition of irrigation
            et_nirr=et-et_if-et_ic-et_wb
            
            # no groundwater routing
            qg[t] = np.nansum(qg_s[t,:])
            #qs.append(np.nansum(v_rs*Pixel_areas_km2/1000)) # MCM
            qsx=np.nansum(v_rs*Pixel_areas_km2/1000) #MCM
            qs[i]=qsx
            
            et_h[t] = et_h[t] + eg_h[t]
            
            #Calculation of green and blue ET 
            etg=v_ed + v_es + v_ei 
            etb=v_eg_spa + et_wb + et_if + et_ic
            et_irr=et_if+et_ic
            
            #replacement of 0 by mintool for ET, so that %tual difference can be calculated 
            ET_act_n0=np.where(ET_act==0, mintol, ET_act)
            et_n0=np.where(et==0, mintol, et)
            et_nirr_n0=np.where(et_nirr==0, mintol, et_nirr)
            
            #calculation for the percential difference between actual ET and ET from WetSpa
            et_round=np.round(et, decimals=2)
            et_perc=np.where(np.logical_or(ET_act == 0, et_round == 0), np.nan, ET_act/et_round)
            #et_perc=ET_act_n0/et_n0
            #et_perc=xr.apply_ufunc(perc2, ET_act_n0, et_n0, dask='allowed')
            et_nirr_round=np.round(et_nirr, decimals=2)
            et_perc_nirr=np.where(np.logical_or(ET_act == 0, et_nirr_round == 0), np.nan, ET_act/et_nirr)
            #et_perc_nirr=ET_act_n0/et_nirr_n0
            #et_perc_nirr=xr.apply_ufunc(perc2, ET_act_n0, et_nirr_n0, dask='allowed')
            
            SRO['time'] = P.time
            ET['time'] = P.time
            ET_diff['time']=P.time
            ET_diff_irr['time']=P.time
            ETb['time']=P.time
            ETg['time']=P.time
            Sup['time']=P.time
            ETi['time']=P.time
            ETs['time']=P.time
            ETd['time']=P.time
            ETgw['time']=P.time
            ETirr['time']=P.time
            SM['time']=P.time
            ET_per['time'] = P.time 
            ET_per_irr['time']=P.time

            SRO=xr.apply_ufunc(sum2, v_rs, zar, dask = 'allowed')
            ET=xr.apply_ufunc(sum2, et, zar, dask='allowed')
            ET_diff=xr.apply_ufunc(dif2, ET_act, et, dask='allowed')
            ET_diff_irr=xr.apply_ufunc(dif2, ET_act, et_nirr, dask='allowed')
            ETg=xr.apply_ufunc(sum2, etg, zar, dask='allowed')
            ETb=xr.apply_ufunc(sum2, etb, zar, dask='allowed')
            Sup=xr.apply_ufunc(sum2, sup, zar, dask='allowed')
            ETi=xr.apply_ufunc(sum2, v_ei, zar, dask='allowed')
            ETs=xr.apply_ufunc(sum2, v_es, zar, dask='allowed')
            ETd=xr.apply_ufunc(sum2, v_ed, zar, dask='allowed')
            ETgw=xr.apply_ufunc(sum2, v_eg_spa, zar, dask='allowed')
            ETirr=xr.apply_ufunc(sum2, et_nirr, zar, dask='allowed')
            SM=xr.apply_ufunc(sum2, soil_moisture, zar, dask='allowed')
            ET_per=xr.apply_ufunc(sum2, et_perc, zar, dask='allowed')
            ET_per_irr=xr.apply_ufunc(sum2, et_perc_nirr, zar, dask='allowed')
            
            if t == 0:
                sro = SRO
                et_new = ET
                et_dif_new=ET_diff
                et_dif_irr_new=ET_diff_irr
                etb_new=ETb
                etg_new=ETg
                sup_new=Sup
                eti=ETi
                ets=ETs
                etd=ETd
                etgw=ETgw
                etirr=ETirr
                sm=SM
                et_per=ET_per
                et_per_irr=ET_per_irr
                
        
            else:
                sro = xr.concat([sro, SRO], dim='time')
                et_new = xr.concat([et_new, ET], dim='time')
                et_dif_new=xr.concat([et_dif_new, ET_diff], dim='time')
                et_dif_irr_new=xr.concat([et_dif_irr_new, ET_diff_irr], dim='time')
                etb_new=xr.concat([etb_new, ETb], dim='time')
                etg_new=xr.concat([etg_new, ETg], dim='time')
                sup_new=xr.concat([sup_new, Sup], dim='time')
                eti=xr.concat([eti, ETi], dim='time')
                ets=xr.concat([ets, ETs], dim='time')
                etd=xr.concat([etd, ETd], dim='time')
                etgw=xr.concat([etgw, ETgw], dim='time')
                etirr=xr.concat([etirr, ETirr], dim='time')
                sm=xr.concat([sm, SM], dim='time')
                et_per=xr.concat([et_per, ET_per], dim='time')
                et_per_irr=xr.concat([et_per_irr, ET_per_irr], dim='time')
                
            del SRO, ET, ET_diff, ET_diff_irr, ETb, ETg, Sup, ETi, ETs, ETd, ETgw, ETirr, SM, ET_per, ET_per_irr
        del lu    
    del LU, Pt
    
    #Surface Runoff 
    if('SRO' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Surface_Runoff_M"}
        sro.attrs=attrs
        sro.name = 'Surface_Runoff_M'
    else:
        sro = None
    
    #Evapotranspiration
    if('ET' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_M"}
        et_new.attrs=attrs
        et_new.name = 'Evapotranspiration_M'
    else:
        et_new = None
    
    #Difference between actual Evapotranspiration and evapotranspiration calculated in WetSpa 
    if('ET_diff' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_difference_M"}
        et_dif_new.attrs=attrs
        et_dif_new.name = 'Evapotranspiration_difference_M'
    else:
        et_dif_new = None
    
    #Difference between actual Evapotranspiration and evapotranspiration calculated in WetSpa -> irrigation not taken into account
    if('ET_diff_noirr' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_difference_no_irrgation_M"}
        et_dif_irr_new.attrs=attrs
        et_dif_irr_new.name = 'Evapotranspiration_difference_no_irrigation_M'
    else:
        et_dif_irr_new = None
    
    #green ET 
    if('ETg' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_green_M"}
        etg_new.attrs=attrs
        etg_new.name = 'Evapotranspiration_green_M'
    else:
        etg_new = None
    
    #blue ET 
    if('ETb' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_blue_M"}
        etb_new.attrs=attrs
        etb_new.name = 'Evapotranspiration_blue_M'
    else:
        etb_new = None
    
    #Suplly
    if('Sup' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Supply_M"}
        sup_new.attrs=attrs
        sup_new.name = 'Supply_M'
    else:
        sup_new = None
        
    #Evapotranspiration from interception 
    if('ETi' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_interception_M"}
        eti.attrs=attrs
        eti.name = 'Evapotranspiration_interception_M'
    else:
        eti = None
    
    #Evapotranspiration from the unsaturated zone 
    if('ETs' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_unsaturated_M"}
        ets.attrs=attrs
        ets.name = 'Evapotranspiration_unsaturated_M'
    else:
        ets = None
    
    #Evapotranspiration from depression 
    if('ETd' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_depression_M"}
        etd.attrs=attrs
        etd.name = 'Evapotranspiration_depression_M'
    else:
        etd = None
    
    #Evapotranspiration from ground water  
    if('ETgw' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_groundwater_M"}
        etgw.attrs=attrs
        etgw.name = 'Evapotranspiration_groundwater_M'
    else:
        etgw = None
    
    #Evapotranspiration from irrigation  
    if('ET_nirr' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_irrigation_M"}
        etirr.attrs=attrs
        etirr.name = 'Evapotranspiration_no_irrigation_M'
    else:
        etirr = None
    
    #Soil moisture 
    if('SM' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Soil_Moisture_M"}
        sm.attrs=attrs
        sm.name = 'Soil_Moisture_M'
    else:
        sm = None
    
    #percentage of ET from WetSpa of actual ET  
    if('ET_per' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_percentage_M"}
        et_per.attrs=attrs
        et_per.name = 'Evapotranspiration_percentage_M'
    else:
        et_per = None
    
    #percentage of ET from WetSpa (without the inclusion of irrgated land uses ) of actual ET  
    if('ET_per_noirr' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Evapotranspiration_percentage_no_irrigation_M"}
        et_per_irr.attrs=attrs
        et_per_irr.name = 'Evapotranspiration_percentage_no_irrigation_M'
    else:
        et_per_irr = None
    
    

    return sro, et_new, et_dif_new, et_dif_irr_new, etg_new, etb_new, sup_new, eti, ets, etd, etgw, sm, et_per, et_per_irr, etirr


