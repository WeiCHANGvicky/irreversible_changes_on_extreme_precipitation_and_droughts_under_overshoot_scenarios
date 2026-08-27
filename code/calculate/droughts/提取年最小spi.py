# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:05:32 2026

@author: WeiCH
"""

import numpy as np
import xarray as xr
import os,glob
import warnings
warnings.filterwarnings('ignore')

files = sorted(glob.glob(r'F:\spi\ssp534\spi_1\*.nc'))

for file in files:
    file_name = os.path.basename(file)
    print(file_name)
    out_dir = r'E:\data\spi\year\ssp534\spi1'
    os.makedirs(out_dir, exist_ok=True)

    
    ds=xr.open_dataset(file)
    spi_3 = ds['spi_3'].sel(time=slice('1950-02-01','2300-12-01'))
    drought =spi_3.groupby('time.year').min('time')
    drought.to_netcdf(fr'{out_dir}\{file_name}')
    print(f'done {file_name}')