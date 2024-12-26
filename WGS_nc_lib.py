#coding:utf-8
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

from typing import Optional, Union, Sequence, Dict
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from einops import rearrange
#from ..augmentation import TransformsFixRotation
#from ...config import cfg
import matplotlib.pyplot as plt  
import sys
import netCDF4 as nc
import cv2
import time
from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num, date2index, Dataset  # http://code.google.com/p/netcdf4-python/

np.set_printoptions(threshold=sys.maxsize)
np_dtype = np.float32 


def plot_var(var,Fname):
	cmap = plt.cm.coolwarm
	ylist = np.linspace(1, var.shape[0], var.shape[0])
	xlist = np.linspace(1, var.shape[1], var.shape[1])
	X, Y = np.meshgrid(xlist, ylist)
	plt.figure()
	#levels = [-0.2,0.0, 0.2]
	levels = np.linspace(-2, 2, 100)
	contour = plt.contour(X, Y, var, colors='k')
	plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
        #plt.imshow(var)
	#contour_filled = plt.contourf(X, Y, var,cmap=cmap)
	contour_filled = plt.contourf(X, Y, var,levels,cmap=cmap)
	plt.colorbar(contour_filled)
	plt.title(Fname)
	plt.xlabel('lon')
	plt.ylabel('lat')
	plt.savefig('ori_'+str(Fname)+'.jpg',dpi=300)
	plt.close()

def readnc(nc_file,Blat,Elat,Blon,Elon,Slev,in_channels,clim):
	#print ("------- read files ------------")
	TS = np.random.rand(len(nc_file),len(Slev),int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	#ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	#ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		#print (nc_file[i])
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if str(in_channels) == '1':
			TS[i,:,:,:] = ori_variables['thetao'][0,Slev,Blat:Elat,Blon:Elon]
		else :
			TS[i,:,:,:] = ori_variables['so'][0,Slev,Blat:Elat,Blon:Elon]

		TS[TS<-100.] = -9999.
		TS[TS>100.] = -9999.
		TS[i,:,:,:] = TS[i,:,:,:]-clim
		TS[TS<-100.] = 0.0
		TS[TS>100.] = 0.0
		ori_data.close()
		#plot_var(TS[i,0,:,:],'TSa'+str(i))

	TS = np.moveaxis(TS, 1, -1)
	return TS



def readnc_SST(nc_file,Blat,Elat,Blon,Elon,SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim):
	#print ("------- read files ------------")
	SST = np.random.rand(len(nc_file),1,int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	#ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	#ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		#print (nc_file[i])
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if str(in_channels) == '1':
			sst20 = ori_variables['analysed_sst'][0,SST_Blat:SST_Elat,SST_Blon:SST_Elon]
			SST[i,0,:,:] = cv2.resize(sst20,(SST.shape[3],SST.shape[2]),interpolation=cv2.INTER_LINEAR)
		else :
			sst20 = ori_variables['analysed_sst'][0,SST_Blat:SST_Elat,SST_Blon:SST_Elon]
			SST[i,0,:,:] = cv2.resize(sst20,(SST.shape[3],SST.shape[2]),interpolation=cv2.INTER_LINEAR)

		SST[i,:,:,:]=SST[i,:,:,:]-273.15
		#SST[SST<-100.] = -9999.
		#SST[SST>100.] = -9999.
		SST[i,:,:,:] = SST[i,:,:,:]-clim[:1,:,:]
		SST[SST<-100.] = 0.0
		SST[SST>100.] = 0.0
		#print (SST[0,0,:,:])
		#exit()
		#plot_var(SST[i,0,:,:],'SSTa'+str(i))
		ori_data.close()

	SST = np.moveaxis(SST, 1, -1)
	return SST


def readnc_SLA(nc_file,Blat,Elat,Blon,Elon,var,SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels):
	#print ("------- read files ------------")
	SLA = np.random.rand(len(nc_file),1,int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	#ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	#ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		#print (nc_file[i])
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if str(in_channels) == '1':
			sla4 = ori_variables[var][0,SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[3],SLA.shape[2]),interpolation=cv2.INTER_LINEAR)
		else :
			sla4 = ori_variables[var][0,SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[3],SLA.shape[2]),interpolation=cv2.INTER_LINEAR)

		#SST[SST<-100.] = -9999.
		#SST[SST>100.] = -9999.
		#SST[i,:,:,:] = SST[i,:,:,:]-clim[:1,:,:]
		SLA[SLA<-100.] = 0.0
		SLA[SLA>100.] = 0.0
		#print (SLA[0,0,:,:])
		#exit()
		#plot_var(SLA[i,0,:,:],var+"a"+str(i))
		ori_data.close()

	SLA = np.moveaxis(SLA, 1, -1)
	return SLA

def write_to_nc_4dvar(data,var_name,file_name_path,dt,Rdep,RBlat, RElat,RBlon, RElon):

    #nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)
    
    
    ntim=data.shape[0]
    nlev=data.shape[1]
    nlat=data.shape[2]
    nlon=data.shape[3]

    
    data[data<-100.] = -9999.
    #matrix=np.arange(out_len)
    lon_list = np.linspace(RBlon, RElon, nlon) 
    lat_list = np.linspace(RBlat, RElat, nlat)   
    lev_list = Rdep
    #lev_list = np.linspace(1, nlev, nlev)
    tim_list = np.linspace(1, ntim, ntim)

    da=nc.Dataset(file_name_path,'w',format='NETCDF4_CLASSIC')


    da.createDimension('depth', nlev)
    da.createDimension('time', ntim)
    da.createDimension('longitude',nlon)  #创建坐标点
    da.createDimension('latitude',nlat)  #创建坐标点


    #da.createVariable("lon",'f',("lons"))  #添加coordinates  'f'为数据类型，不可或缺
    #da.createVariable("lat",'f',("lats"))  #添加coordinates  'f'为数据类型，不可或缺

    # Create coordinate variables for 4-dimensions
    #times = da.createVariable('time', np.int32, ('time',)) 
    depths =da.createVariable('depth', np.float32, ('depth',)) 
    latitudes  = da.createVariable('latitude', np.float32,('latitude',))
    longitudes = da.createVariable('longitude', np.float32,('longitude',))
    #times      = da.createVariable('time', nc_fid.variables['time'].dtype,('time',))
    times      = da.createVariable('time', np.float32,('time',))
    # You can do this step yourself but someone else did the work for us.
    #for ncattr in nc_fid.variables['time'].ncattrs():
    #     times.setncattr(ncattr, nc_fid.variables['time'].getncattr(ncattr))



    # Global Attributes
    da.description = var_name+' Forecast by Deep Learning, National Marine Data and Information Servace (MEIT) WGS' 
    da.history = 'Created ' + time.ctime(time.time()) 
    da.source = 'netCDF4 python module tutorial'
    # Variable Attributes 
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    depths.units = 'm'
    #times.units = 'hours since 0001-01-01 00:00:00'
    Ydt = datetime.strptime(dt, "%Y%m%d")
    Sdt = (Ydt + timedelta(days=-1)).strftime("%Y-%m-%d")
    times.units = 'days since '+str(Sdt)
    times.calendar = 'gregorian'
    # Create the actual 4-d variable
    da.createVariable(var_name, np.float32,('time','depth','latitude','longitude'),fill_value=-9999.)
    #da.createVariable('u', np.float32,('time','lat','lon'))
    da._FillValue = -9999.


    da.variables['latitude'][:]=lat_list     #填充数据
    da.variables['longitude'][:]=lon_list     #填充数据
    #da.variables['time'][:]=time     #填充数据
    # Assign the dimension data to the new NetCDF file.
    da.variables['time'][:] = tim_list
    da.variables['depth'][:]=lev_list     #填充数据
    
    ##da.createVariable('u','f8',('lats','lons')) #创建变量，shape=(627,652)  'f'为数据类型，不可或缺
    da.variables[var_name][:]=data       #填充数据
    da.close()

def readnc_var(nc_file,Blat,Elat,Blon,Elon,var,SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,Tdvar=False):
	SLA = np.random.rand(len(nc_file),1,int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if Tdvar:
			sla4 = ori_variables[var][0,SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[3],SLA.shape[2]),interpolation=cv2.INTER_LINEAR)
		else :
			sla4 = ori_variables[var][SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[3],SLA.shape[2]),interpolation=cv2.INTER_LINEAR)

		#SST[SST<-100.] = -9999.
		#SST[SST>100.] = -9999.
		#SST[i,:,:,:] = SST[i,:,:,:]-clim[:1,:,:]
		SLA[SLA<-100.] = 0.0
		SLA[SLA>100.] = 0.0
		#print (SLA[0,0,:,:])
		#exit()
		#plot_var(SLA[i,0,:,:],var+"a"+str(i))
		ori_data.close()

	SLA = np.moveaxis(SLA, 1, -1)
	return SLA


if __name__ == "__main__":
    main()
