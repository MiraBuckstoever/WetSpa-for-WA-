# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import datetime
import numpy as np
import xarray as xr
import glob
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
'E'
os.chdir('E:\Master\Master_thesis\WetSpa-Python_v1_0\codes')
import WetSpa_main_model_fuctions_monte as WetSpa
import WetSpa_static_preproc as WetSpa_prepros


def mont_carlo(n):
    #definitions that are only needed once 
    client = WetSpa.start_multiprocessing()
    client.restart()
    
    df = pd.DataFrame(columns=['k_ep', 'k_run', 'P_max', 'k_ss', 'ki', 'dt_hours', 'kg', 'g0', 'gmax'])
    
    MAIN_FOLDER = r'E:\Master\Master_thesis\WetSpa-Python_v1_0'
    time_now = datetime.datetime.now()
    time_str = time_now.strftime('%Y_%m_%d_%Hh_%Mm')
    output_dir = 'output_dir_'+str(time_str)
    dir_out = os.path.join(MAIN_FOLDER,'output', 'nc', output_dir)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    log_file_path = os.path.join(dir_out, 'log_file.txt')
    log_file = WetSpa._log_create(log_file_path)
    output = ['SRO', 'Pt']
    input_files =  { # SB! sub-catchment
                'p_in' : os.path.join(MAIN_FOLDER,"input/maps/p_CHIRPS_monthly.nc")
                }
    log_file_path = os.path.join(dir_out, 'log_file.txt')
    log_file = WetSpa._log_create(log_file_path)
    latchunk = 200
    lonchunk = 200
    timechunk = 1
    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk}

    comp = dict(zlib=True, complevel=9, least_significant_digit=2, chunksizes=list(chunks.values()))  

    encoding_output = dict(
            dtype =  np.float32,
            _FillValue = np.nan,
            scale_factor = np.float32(1.0),
            add_offset = np.float32(0.0),
            zlib = True,
            complevel = 9,
            least_significant_digit = 2,
            chunksizes=tuple(chunks.values())
        )
    
    def nc2ts(bf_sr, shape):
        ts_all=[]
        for nc in bf_sr: #insert paths to your BaseFlow and SRO output netCDF files
            var,name = WetSpa.open_nc(nc,chunks)
            print('*',name)   #nothing is printed as this is only loading the function

            #Clip the datset by shape of the subbasins 
            var = var.rio.set_crs("epsg:4326", inplace=True)
            var_clipped = var.rio.clip(shape.geometry.values, shape.crs, drop=False)

            # Calculate area of the subbasins
            area = WetSpa.area_sqkm(var_clipped[0])

            # compute volume by multiplying the depth by area
            Volume=var_clipped*area*1e-6
            Volume = Volume.rename('{0}'.format(var.name))
            # sum the volume spatially and convert it dataframe to have timeseries values
            ts = Volume.sum(dim=['latitude','longitude'], skipna=True).to_dataframe()
            ts_all.append(ts)
            var.close()
            var_clipped.close()
        return ts_all
    
    #number of monte carlo computations 
    for i in range(n): 
        k_ep=1.5#np.random.uniform(1,3)
        k_run=1#np.random.uniform(1,2)
        P_max=20.000000#np.random.uniform(1,2)
        k_ss=1.0542748#np.random.uniform(1,2)
        ki=50.00#np.random.uniform(1,2)
        dt_hours=720#np.random.uniform(1,2)
        kg=0.009782#np.random.uniform(1,2)
        g0=50.00#np.random.uniform(1,2)
        g_max=282.46124#np.random.uniform(1,2)
        #next two lines need to be further down in the code 
        row =pd.DataFrame({'k_ep':[k_ep], 'k_run':[k_run], 'P_max':[P_max], 'k_ss':[k_ss], 'ki':[ki], 'dt_hours':[dt_hours], 'kg':[kg], 'g0':[g0], 'gmax':[g_max]})
        df=pd.concat([df, row], ignore_index=True)
        
        #assignments of inputs that are needed for the WetSpa model 
        
        parameters = {
            'k_ep':k_ep,
            'k_run':k_run,
            'P_max':P_max,
            'k_ss':k_ss,
            'ki':ki,
            'dt_hours':dt_hours,
            'kg':kg,
            'g0':g0,
            'g_max':g_max
            }
        wetspa_params={
            'output':output,
            'parameters':parameters
            }
        result=WetSpa.wetspa_model(wetspa_params)
        print(result)
        print(type(result))
        #write output as netcdf files
        from dask.distributed import progress
        import time
        t1 = time.perf_counter () #returns value (in fractional seconds) of a performance counter 
        log_file.write('{:>26s}:\n'.format('Output files path'))
        print("writing the netcdf file")   
        for r in result: #outputs that were previously selected 
            if(r is not None):
                #print(r)
                print("* {0}".format(r.name))
                nc_fn=r.name+'.nc'
                nc_path=os.path.join(dir_out,nc_fn)
                encoding = {r.name: encoding_output}
        
#         r.to_netcdf(nc_path,encoding=encoding)
#         r.close()

        delayed_obj = r.to_netcdf(nc_path,encoding=encoding, compute=True)
        result = delayed_obj#.persist()
        progress(result)
        r.close()
        name = r.name
        log_file.write('{:>26s}: {}\n'.format(name, str(nc_path)))
        # del r
        print(f"Writing the netCDF is completed!")
        #why is the time important in this step? Time was not measured/taken into consideration for the other steps 
        t2 = time.perf_counter ()
        time_diff = t2 - t1
        print(f"\nIt took {time_diff} Secs to execute this method")
        WetSpa._log_close(log_file)
        
        #Reading of the files 
        infiles = [input_files['p_in']]#,input_files['e_in']
        outfiles = glob.glob(os.path.join(dir_out,'*.nc'))

        #files_to_read = ['Base_flow','Surface_Runoff','Incremental_ET_M','Rainfall_ET_M']
        files_to_read = ['Surface_Runoff']#'Base_flow',
        x = pd.Series(outfiles)
        bf_sr = x[x.str.contains('|'.join(files_to_read))]
        bf_sr = infiles+bf_sr.to_list()
        
        #Calculations for KGE and NSE that are needed to define later which parameters are best suited 
        print('KGE and NSE calculation for Lolsur')  
        shapef = os.path.join(MAIN_FOLDER,"input\shapefile\Lolsur_basin.shp")
        shape = gpd.read_file(shapef,crs="EPSG:4326")
        shape = shape.to_crs("EPSG:4326")
        ts_all = nc2ts(bf_sr, shape)
        df = pd.concat(ts_all, axis =1)  
        # read observed outflow
        path_obs = os.path.join(MAIN_FOLDER,"input/csvs/discharge_Lolsur_monthly.csv")
        Q_m=pd.read_csv(path_obs,sep=',',index_col=0,skiprows=0)
        Q_m.index=[datetime.datetime.strptime(y,'%d/%m/%Y') for y in Q_m.index]#definition of time was different bevor but gave errors 
        Q_m = Q_m[df.index[0]:df.index[-1]]
        #read additional inflow for K3 
        path_obs = os.path.join(MAIN_FOLDER,"input/csvs/inflow_K3_monthly.csv")
        Q_m_K3=pd.read_csv(path_obs,sep=',',index_col=0,skiprows=0)
        Q_m_K3.index=[datetime.datetime.strptime(y,'%d/%m/%Y') for y in Q_m_K3.index]
        Q_m_K3 = Q_m_K3[df.index[0]:df.index[-1]]

        df2= pd.concat([df, Q_m, Q_m_K3], axis =1)
        NSE = WetSpa.nash_sutcliffe(df2['Lolsur_Q_in_km3/month'], df2['Surface_Runoff_M'] + df2['K3_Q_in_km3/month'])
        df3 = df2.dropna()
        KGE = WetSpa.kge(df3['Lolsur_Q_in_km3/month'], df3['Surface_Runoff_M']+df3['K3_Q_in_km3/month'])
        print(NSE)
        print(KGE)


z=mont_carlo(1)

