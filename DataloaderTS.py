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
#import WGS_nc_lib
from WGS_nc_lib import plot_var,readnc_SLA,readnc_SST,readnc,readnc_var
#from netCDF4 import num2date, date2num, date2index, Dataset  # http://code.google.com/p/netcdf4-python/

np.set_printoptions(threshold=sys.maxsize)

#default_data_dir = os.path.join(cfg.datasets_dir, "earthnet2021")

default_data_dir = "/home/wang/145_SRRAM_SCS/3DTS-Project3/Unet-TSformer/SCS_test/SST_WIND/DATA/CMEMS/"
default_sst_dir  = "/home/wang/145_SRRAM_SCS/3DTS-Project3/Unet-TSformer/SCS_test/SST_WIND/DATA/OSTIA/"
default_sla_dir  = "/home/wang/145_SRRAM_SCS/3DTS-Project3/Unet-TSformer/SCS_test/SST_WIND/DATA/AVISO/"
default_wnd_dir  = "/home/wang/145_SRRAM_SCS/3DTS-Project3/Unet-TSformer/SCS_test/SST_WIND/DATA/WIND/"
#default_data_dir = "/ooi_data/WGS_DATA/Mercatorglorys12v1/Mercator_glorys1v12_2019"
#default_sst_dir = "/ooi_data/WGS_DATA/SST/OSTIA/OSTIA_2019"
#default_sla_dir = "/home/wind/2Project3-MEIT/NRT_download/AVISO/DT_DATA/2019"
print (default_data_dir)

dxdy=12
SST_dxdy = 20
WND_dxdy = 8
SLA_dxdy = 4
sla_scale = 4.0

lato=2
lono=105
nlat=312
nlon=240

lato=17
lono=117
nlat=96
nlon=96

lato=22
lono=124
nlat=72
nlon=72

lato=22
lono=121
nlat=72
nlon=72

lato=20
lono=120
nlat=120
nlon=120

lato=0
lono=100
nlat=360
nlon=360

lato=5
lono=105
nlat=240
nlon=240

Tint = WGS_Tint
in_len = WGS_in_len
out_len = WGS_out_len
in_channels = WGS_in_channels
ot_channels = 1
#Slev=[0,4,7,10,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
#Slev=[0,7,12,14,16,17,20,21,23,24,25,26]
#Slev=[0,7,12,14,16,17]
#Slev=[0]
# 0.5 9.5 50 
Slev=[0,7,17]

#lev = 10
#Slev=[4,7,10,12,13,14,15,16,17,20]
#lev = 15
#Slev=[4,7,10,12,13,14,15,16,17,20,21,22,23,24,25]
#lev = 5
#Slev=[4,7,10,12,13]
#data_channels = 1
#Anum = 1000


ABlat=(80+lato)*dxdy
AElat=ABlat+nlat
ABlon=(180+lono)*dxdy
AElon=ABlon+nlon

Blat=(lato-0)*dxdy
Elat=Blat+nlat
Blon=(lono-100)*dxdy
Elon=Blon+nlon

SST_Blat=(90+lato)*SST_dxdy
SST_Elat=int(SST_Blat+(nlat*SST_dxdy/12))
SST_Blon=(180+lono)*SST_dxdy
SST_Elon=int(SST_Blon+(nlon*SST_dxdy/12))

SLA_Blat=(90+lato)*SLA_dxdy
SLA_Elat=int(SLA_Blat+(nlat*SLA_dxdy/12))
SLA_Blon=(0+lono)*SLA_dxdy
#SLA_Blon=(180+lono)*SLA_dxdy
SLA_Elon=int(SLA_Blon+(nlon*SLA_dxdy/12))

WND_Blat=(lato+1)*WND_dxdy
WND_Elat=int(WND_Blat+(nlat*WND_dxdy/12))
WND_Blon=(lono-99)*WND_dxdy
WND_Elon=int(WND_Blon+(nlon*WND_dxdy/12))


Tin_len=in_len*Tint
Tout_len=out_len*Tint


height = Elon-Blon
width = Elat-Blat
#height = Elat-Blat
#width = Elon-Blon

np_dtype = np.float32 
def_value = 0.0
SST_MAX=1.5
WND_MAX=10
curl_MAX=1e-6
SST_MIN=-1.5

def scale_sst(sst):
    return (sst - SST_MIN) / (SST_MAX - SST_MIN)

def scale_back_sst(sst):
    return (SST_MAX - SST_MIN) * sst + SST_MIN

clim = np.random.rand(len(Slev),int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
ori_data  = nc.Dataset('mercatorglorys12v1_gl12_mean_1993-2012.nc')     # 读取nc文件
ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
if str(in_channels) == '1':
  clim = ori_variables['thetao'][0,Slev,ABlat:AElat,ABlon:AElon]
else :
  clim = ori_variables['so'][0,Slev,ABlat:AElat,ABlon:AElon]
ori_data.close()

clim[clim<-100.] = -9999.
clim[clim> 100.] = -9999.
#clim=np.nan_to_num(clim)
clim = clim.filled(-9999.)
'''
print (clim.shape)
print (np.max(clim))
print (np.min(clim))
print (clim[lev-1,:,0])
print (clim[0,:,0])
'''
#/home/wind/2Project3-MEIT/NRT_download/AVISO/DT_DATA/train

"""Code is adapted from https://github.com/tychovdo/MovingMNIST."""

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import torch.nn.functional as F
import codecs
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader
from typing import Optional
'''
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

def readnc(nc_file,Blat,Elat,Blon,Elon):
	#print ("------- read files ------------")
	TS = np.random.rand(len(nc_file),lev,width,height).astype(np_dtype)
	ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
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



def readnc_SST(nc_file,Blat,Elat,Blon,Elon):
	#print ("------- read files ------------")
	SST = np.random.rand(len(nc_file),1,int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		#print (nc_file[i])
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if str(in_channels) == '1':
			sst20 = ori_variables['analysed_sst'][0,SST_Blat:SST_Elat,SST_Blon:SST_Elon]
			SST[i,0,:,:] = cv2.resize(sst20,(SST.shape[2],SST.shape[3]),interpolation=cv2.INTER_LINEAR)
		else :
			sst20 = ori_variables['analysed_sst'][0,SST_Blat:SST_Elat,SST_Blon:SST_Elon]
			SST[i,0,:,:] = cv2.resize(sst20,(SST.shape[2],SST.shape[3]),interpolation=cv2.INTER_LINEAR)

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


def readnc_SLA(nc_file,Blat,Elat,Blon,Elon,var):
	#print ("------- read files ------------")
	SLA = np.random.rand(len(nc_file),1,int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
	ori_data  = nc.Dataset(nc_file[0])     # 读取nc文件
	ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
	for i in range(len(nc_file)):
		ori_data  = nc.Dataset(nc_file[i])     # 读取nc文件
		#print (nc_file[i])
		ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
		if str(in_channels) == '1':
			sla4 = ori_variables[var][0,SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[2],SLA.shape[3]),interpolation=cv2.INTER_LINEAR)
		else :
			sla4 = ori_variables[var][0,SLA_Blat:SLA_Elat,SLA_Blon:SLA_Elon]
			SLA[i,0,:,:] = cv2.resize(sla4,(SLA.shape[2],SLA.shape[3]),interpolation=cv2.INTER_LINEAR)

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
'''

def change_layout(data: np.ndarray,
                  in_layout: str = "HWCT",
                  out_layout: str = "HWCT"):
    axes = [None, ] * len(in_layout)
    for i, axis in enumerate(in_layout):
        axes[out_layout.find(axis)] = i
    return np.transpose(data, axes=axes)

def einops_change_layout(data: Union[np.ndarray, torch.Tensor],
                         einops_in_layout: str = "H W C T",
                         einops_out_layout: str = "H W C T"):
    return rearrange(data, f"{einops_in_layout} -> {einops_out_layout}")

class _BaseEarthNet2021Dataset(Dataset):
    r"""
    An .npy file contains a dict with

    "highresdynamic":   np.ndarray
        shape = (128, 128, 7, T_highres)
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, T_meso)
        channels are [precipitation, pressure, temp mean, temp min, temp max]
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]

    train:
        T_highres = 30, T_meso = 150
    iid/ood test:
        T_highres = 10, T_meso = 150 for context
        T_highres = 20 for target
    extreme test:
        T_highres = 20, T_meso = 300 for context
        T_highres = 40 for target
    seasonal test:
        T_highres = 70, T_meso = 1050 for context
        T_highres = 140 for target
    """
    default_layout = "HWCT"
    default_static_layout = "HWC"
    # flip requires the last two dims to be H,W
    layout_for_aug = "CTHW"
    static_layout_for_aug = "CHW"

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 layout: str = default_layout,
                 static_layout: str = default_static_layout,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False, ):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        data_aug_mode:  str
            If None, no data augmentation is performed
            If "0", apply `RandomHorizontalFlip(p=0.5)` and RandomVerticalFlip(p=0.5)
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        self.return_mode = return_mode
        self.data_aug_mode = data_aug_mode
        if data_aug_cfg is None:
            data_aug_cfg = {}
        self.data_aug_cfg = data_aug_cfg
        if self.data_aug_mode is None:
            pass
        elif self.data_aug_mode in ["0", ]:
            self.data_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])
        elif self.data_aug_mode in ["1", "2"]:
            self.data_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                #TransformsFixRotation([0, 90, 180, 270]),
            ])
        else:
            raise NotImplementedError
        if layout is None:
            layout = self.default_layout
        self.layout = layout
        if static_layout is None:
            static_layout = self.default_static_layout
        self.static_layout = static_layout
        self.highresstatic_expand_t = highresstatic_expand_t
        self.mesostatic_expand_t = mesostatic_expand_t
        if isinstance(meso_crop, str):
            assert meso_crop == "default", f"meso_crop mode {meso_crop} not supported."
            meso_crop = ((39, 41), (39, 41))
        self.meso_crop = meso_crop
        self.np_dtype = np.float16 if fp16 else np.float32

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_default_static_layout(self):
        if not hasattr(self, "_einops_default_static_layout"):
            self._einops_default_static_layout = " ".join(self.default_static_layout)
        return self._einops_default_static_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    @property
    def einops_static_layout(self):
        if not hasattr(self, "_einops_static_layout"):
            self._einops_static_layout = " ".join(self.static_layout)
        return self._einops_static_layout

    @property
    def einops_layout_for_aug(self):
        if not hasattr(self, "_einops_layout_for_aug"):
            self._einops_layout_for_aug = " ".join(self.layout_for_aug)
        return self._einops_layout_for_aug

    @property
    def einops_static_layout_for_aug(self):
        if not hasattr(self, "_einops_static_layout_for_aug"):
            self._einops_static_layout_for_aug = " ".join(self.static_layout_for_aug)
        return self._einops_static_layout_for_aug

    def process_raw_data_from_npz(
            self,
            highresdynamic, highresstatic,
            mesodynamic, mesostatic):
        r"""
        Process np.ndarray data loaded from saved .npz files

        Parameters
        ----------
        highresdynamic:   np.ndarray
            shape = (128, 128, 7, T_highres)
            channels are [blue, green, red, nir, cloud, scene, mask]
        highresstatic:    np.ndarray
            shape = (128, 128, 1)
            channel is [elevation]
        mesodynamic:      np.ndarray
            shape = (80, 80, 5, T_meso)
            channels are [precipitation, pressure, temp mean, temp min, temp max]
        mesostatic:       np.ndarray
            shape = (80, 80, 1)
            channel is [elevation]

            (T_highres, T_meso) =
                (30, 150) for train, iid, ood
                (60, 300) for extreme
                (210, 1050) for seasonal

        Returns
        -------
        ret:    dict
        """
        T_highres = highresdynamic.shape[-1]
        T_meso = mesodynamic.shape[-1]

        if self.return_mode in ["default", ]:
            if self.meso_crop is not None:
                mesodynamic = self.crop_meso_spatial(mesodynamic)
                mesostatic = self.crop_meso_spatial(mesostatic)

            highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
            mesodynamic = np.nan_to_num(mesodynamic, copy=False, nan=0.0)
            highresstatic = np.nan_to_num(highresstatic, copy=False, nan=0.0)
            mesostatic = np.nan_to_num(mesostatic, copy=False, nan=0.0)

            highresdynamic = einops_change_layout(
                data=highresdynamic,
                einops_in_layout=self.einops_default_layout,
                einops_out_layout=self.einops_layout)
            mesodynamic = einops_change_layout(
                data=mesodynamic,
                einops_in_layout=self.einops_default_layout,
                einops_out_layout=self.einops_layout)
            if self.highresstatic_expand_t:
                highresstatic = np.repeat(highresstatic[..., np.newaxis],
                                          repeats=T_highres,
                                          axis=-1)
                highresstatic = einops_change_layout(
                    data=highresstatic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            else:
                highresstatic = einops_change_layout(
                    data=highresstatic,
                    einops_in_layout=self.einops_default_static_layout,
                    einops_out_layout=self.einops_static_layout)
            if self.mesostatic_expand_t:
                mesostatic = np.repeat(mesostatic[..., np.newaxis],
                                       repeats=T_meso,
                                       axis=-1)
                mesostatic = einops_change_layout(
                    data=mesostatic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            else:
                mesostatic = einops_change_layout(
                    data=mesostatic,
                    einops_in_layout=self.einops_default_static_layout,
                    einops_out_layout=self.einops_static_layout)
            if self.return_mode == "default":
                if self.data_aug_mode is not None:
                    # TODO: augment all components consistently
                    raise NotImplementedError
                #return {
                #    "highresdynamic": highresdynamic,
                #    "highresstatic": highresstatic,
                #    "mesodynamic": mesodynamic,
                #    "mesostatic": mesostatic,
                #}
                #print (highresdynamic.shape)
                #exit()
                return {"highresdynamic": highresdynamic,}
            else:
                raise NotImplementedError
        elif self.return_mode in ["minimal", ]:
            # only RGB, infrared channels and mask
            highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)

            if self.data_aug_mode is not None:
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout_for_aug)
                highresdynamic = self.data_aug(torch.from_numpy(highresdynamic))
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_layout_for_aug,
                    einops_out_layout=self.einops_layout).numpy()
            else:
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            return {"highresdynamic": highresdynamic,}
        else:
            raise NotImplementedError(f"return_mode {self.return_mode} not supported!")

    def crop_meso_spatial(self, meso_data):
        r"""
        Crop the meso data along spatial dims, under default layout.
        """
        if self.meso_crop is None:
            return meso_data
        else:
            return meso_data[
                   self.meso_crop[0][0]:self.meso_crop[0][1],
                   self.meso_crop[1][0]:self.meso_crop[1][1],
                   ...]
class EarthNet2021TrainDataset(_BaseEarthNet2021Dataset):
    r"""
    An .npy file contains a dict with
    "highresdynamic":   np.ndarray
        shape = (128, 128, 7, 30)
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, 150)
        channels are [precipitation, pressure, temp mean, temp min, temp max]
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]
    """
    default_train_dir = os.path.join(default_data_dir, "train")
    default_sst_train_dir = os.path.join(default_sst_dir, "train")
    default_sla_train_dir = os.path.join(default_sla_dir, "train")
    default_wnd_train_dir = os.path.join(default_wnd_dir, "train")

    T_highres = 30
    T_meso = 150

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 data_dir: Union[Path, str] = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        data_aug_mode:  str
            If None, no data augmentation is performed
            If "0", apply `RandomHorizontalFlip(p=0.5)` and RandomVerticalFlip(p=0.5)
        data_aug_cfg:   dict
            dict which contains cfgs for controlling data augmentation
        data_dir:   Union[Path, str]
            Save dir of training data.
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        super(EarthNet2021TrainDataset, self).__init__(
            return_mode=return_mode,
            data_aug_mode=data_aug_mode,
            data_aug_cfg=data_aug_cfg,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16, )
        if data_dir is None:
            data_dir = self.default_train_dir
            sst_data_dir = self.default_sst_train_dir
            sla_data_dir = self.default_sla_train_dir
            wnd_data_dir = self.default_wnd_train_dir

        self.data_dir = Path(data_dir)
        self.sst_data_dir = Path(sst_data_dir)
        self.wnd_data_dir = Path(wnd_data_dir)
        self.sla_data_dir = Path(sla_data_dir)
        #print ("self.data_dir = ",self.data_dir)
        self.Anpz_path_list = sorted(list(self.data_dir.glob("SCS_Mercator12_T_daily_*.nc")))
        self.Asst_path_list = sorted(list(self.sst_data_dir.glob("*.nc")))
        self.Awnd_path_list = sorted(list(self.wnd_data_dir.glob("*.nc")))
        self.Asla_path_list = sorted(list(self.sla_data_dir.glob("*.nc")))

        self.npz_path_list = sorted(list(self.data_dir.glob("SCS_Mercator12_T_daily_*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.sst_path_list = sorted(list(self.sst_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.wnd_path_list = sorted(list(self.wnd_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.sla_path_list = sorted(list(self.sla_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        print ("train number: ",len(self.npz_path_list))

        if len(self.npz_path_list) == len(self.sst_path_list) == len(self.sla_path_list) == len(self.wnd_path_list):
           print("所有列表的长度相同")
           print ("self.npz_path_list = ",self.npz_path_list[10])
           print ("self.sst_path_list = ",self.sst_path_list[10])
           print ("self.sla_path_list = ",self.sla_path_list[10])
           print ("self.wnd_path_list = ",self.wnd_path_list[10])
        else:
           print("列表的长度不相同，请检查并退出")
           exit()
        #print ("npz_path_list = ",len(self.npz_path_list))

    def __len__(self) -> int:
        return len(self.npz_path_list)

    def __getitem__(self, idx: int) -> dict:
        #data_npz = np.load(self.npz_path_list[idx])
        #print ("=======",str(idx))
        #print ("self.npz_path_list = ",self.npz_path_list[idx])
        #print ("self.sst_path_list = ",self.sst_path_list[idx])
        #print ("self.sla_path_list = ",self.sla_path_list[idx])


        TS = readnc(self.Anpz_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 
        TS_Pes = readnc(self.Anpz_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 
        #print (self.Anpz_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint])
        #print (self.Anpz_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1])
        #exit()
        #Laux_data = TS[:,:,:,:4].copy()
        Laux_data = np.random.rand(TS.shape[0],TS.shape[1],TS.shape[2],8).astype(np_dtype)
        Laux_data[:,:,:,0] = readnc_var(self.Awnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'uwnd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,1] = readnc_var(self.Awnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'vwnd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,2] = readnc_var(self.Awnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'wspd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,3] = readnc_var(self.Awnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'stress_curl',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/curl_MAX 
        Laux_data[:,:,:,4] = readnc_SST(self.Asst_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,
                                        SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 
        Laux_data[:,:,:,5] = readnc_SLA(self.Asla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'sla',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype)*sla_scale
        Laux_data[:,:,:,6] = readnc_SLA(self.Asla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'ugos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 
        Laux_data[:,:,:,7] = readnc_SLA(self.Asla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'vgos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 

        SST_Pes = readnc_SST(self.Asst_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,
                             SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 
        und_Pes = readnc_var(self.Awnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'uwnd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        vnd_Pes = readnc_var(self.Awnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'vwnd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        spd_Pes = readnc_var(self.Awnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'wspd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        cul_Pes = readnc_var(self.Awnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'stress_curl',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/curl_MAX 
        
        for i in range(in_len+out_len):
              TS[i,:,:,:]=TS[i,:,:,:]-TS_Pes[0,:,:,:]
              Laux_data[i,:,:,0]=Laux_data[i,:,:,0]-und_Pes[0,:,:]
              Laux_data[i,:,:,1]=Laux_data[i,:,:,1]-vnd_Pes[0,:,:]
              Laux_data[i,:,:,2]=Laux_data[i,:,:,2]-spd_Pes[0,:,:]
              Laux_data[i,:,:,3]=Laux_data[i,:,:,3]-cul_Pes[0,:,:]
              Laux_data[i,:,:,4]=Laux_data[i,:,:,4]-SST_Pes[0,:,:]
              #plot_var(TS[i,:,:,0],'TS'+str(i))
              #plot_var(Laux_data[i,:,:,0],'uwnd'+str(i))
              #plot_var(Laux_data[i,:,:,1],'vwnd'+str(i))
              #plot_var(Laux_data[i,:,:,2],'wspd'+str(i))
              #plot_var(Laux_data[i,:,:,3],'curl'+str(i))
              #plot_var(Laux_data[i,:,:,4],'SST'+str(i))
              #plot_var(Laux_data[i,:,:,5],'ugos'+str(i))
              #plot_var(Laux_data[i,:,:,6],'vgos'+str(i))
              #plot_var(Laux_data[i,:,:,7],'SLA'+str(i))
        #print (TS[:,0,0,0])
        #print ("--------wnd----------")
        #print (Laux_data[0,:,0,1])
        #print ("--------curl----------")
        #print (Laux_data[0,:,0,3])
        #exit()
        #SST = readnc_SST(self.Asst_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],SST_Blat,SST_Elat,SST_Blon,SST_Elon).astype(np_dtype) 
        #print (TS[:,0,0,0]) 
        #print ("min max = ",np.min(TS),np.max(TS))
        #print ("min max = ",np.min(Laux_data[:,:,:,0]),np.max(Laux_data[:,:,:,0]))
        #print ("min max = ",np.min(Laux_data[:,:,:,1]),np.max(Laux_data[:,:,:,1]))
        #print ("min max = ",np.min(Laux_data[:,:,:,2]),np.max(Laux_data[:,:,:,2]))
        #print ("min max = ",np.min(Laux_data[:,:,:,3]),np.max(Laux_data[:,:,:,3]))
        return {"highresdynamic": TS,"auxiliary": Laux_data,}
        '''
        TS = np.zeros((in_len+out_len,lev,width,height)).astype(np_dtype)
        for i in range(in_len+out_len):
                #print (i)
                #print ("self.npz_path_list = ",self.npz_path_list[idx+i])
                ori_data  = nc.Dataset(self.Anpz_path_list[idx+i])     # 读取nc文件
                ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
                if str(in_channels) == '1':
                        TS[i,:,:,:] = ori_variables['thetao'][0,Slev,Blat:Elat,Blon:Elon]
                else :
                        TS[i,:,:,:] = ori_variables['so'][0,Slev,Blat:Elat,Blon:Elon]
                #TS[TS<-100.] = -9999.
                #TS[TS>100.] = -9999.
                #TS[i,:,:,:] = TS[i,:,:,:]-clim
                #TS[TS<-100.] = 0.0
                #TS[TS>100.] = 0.0
                ori_data.close()

        TS = np.moveaxis(TS, 1, -1)
        '''
        #Tthetao = np.zeros((2,32,32,4)).astype(self.np_dtype)
        #thetao = np.random.rand(2,32,32,4).astype(self.np_dtype)
        #print ("=====================")
        #print (TS[0,:,:,:])
        #thetao = thetao+Tthetao
        #print ("=====================")
        #print (TS)
        #exit()
        #print (thetao.type())
        #print ("thetao.shape = ",thetao.shape)
        #exit()
        #plot_var(thetao[0,:,:,0],'OBS_U')
        #exit()
        #highresdynamic = np.nan_to_num(thetao, copy=False, nan=0.0)
        #highresdynamic = np.nan_to_num(thetao, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        #highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
        '''
        print ("thetao.shape = ",thetao.shape)
        #exit()
        #print ("input file = ",self.npz_path_list[idx])

        # keep only [blue, green, red, nir, mask] channels
        highresdynamic = data_npz["highresdynamic"].astype(self.np_dtype)[:, :, [0, 1, 2, 3, 6], :]
        highresstatic = data_npz["highresstatic"].astype(self.np_dtype)
        mesodynamic = data_npz["mesodynamic"].astype(self.np_dtype)
        mesostatic = data_npz["mesostatic"].astype(self.np_dtype)
        print ("highresdynamic.shape = ",highresdynamic.shape)
        exit()
        #print ("highresstatic.shape = ",highresstatic.shape)
        #print ("mesodynami.shape = ",mesodynamic.shape)
        #print ("mesostatic.shape = ",mesostatic.shape)
        thetao = np.random.rand(30,128,128,5).astype(self.np_dtype)
        if self.data_aug_mode in ["2", ]:
            print ("2222222")
            processed_0 = self.process_raw_data_from_npz(
                highresdynamic, highresstatic,
                mesodynamic, mesostatic)
            data_npz_1 = np.load(self.npz_path_list[np.random.randint(len(self))])
            # keep only [blue, green, red, nir, mask] channels
            highresdynamic_1 = data_npz_1["highresdynamic"].astype(self.np_dtype)[:, :, [0, 1, 2, 3, 6], :]
            highresstatic_1 = data_npz_1["highresstatic"].astype(self.np_dtype)
            mesodynamic_1 = data_npz_1["mesodynamic"].astype(self.np_dtype)
            mesostatic_1 = data_npz_1["mesostatic"].astype(self.np_dtype)
            processed_1 = self.process_raw_data_from_npz(
                highresdynamic_1, highresstatic_1,
                mesodynamic_1, mesostatic_1)
            alpha = self.data_aug_cfg.get("mixup_alpha", 0.2)
            lam = np.random.beta(a=alpha, b=alpha)
            if self.return_mode in ["default", ]:
                ret = {}
                for key, val in processed_0.items():
                    ret[key] = lam * val + (1 - lam) * processed_1[key]
            elif self.return_mode in ["minimal", ]:
                ret = lam * processed_0 + (1 - lam) * processed_1
            else:
                raise NotImplementedError
            return ret
        else:
            #print ("1111111112222222")
            return self.process_raw_data_from_npz(
                highresdynamic, highresstatic,
                mesodynamic, mesostatic)
         '''

class EarthNet2021TestDataset(_BaseEarthNet2021Dataset):
    r"""
    An .npy file contains a dict with
    "highresdynamic":   np.ndarray
        shape = (128, 128, 5, T_highres_context) for context
        shape = (128, 128, 5, T_highres_target) for target
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, T_meso) for context
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]
    """
    default_test_dir = os.path.join(default_data_dir, "test")
    default_sst_test_dir = os.path.join(default_sst_dir, "test")
    default_wnd_test_dir = os.path.join(default_wnd_dir, "test")
    default_sla_test_dir = os.path.join(default_sla_dir, "test")

    default_iid_test_data_dir = os.path.join(default_data_dir, "test")
    default_ood_test_data_dir = os.path.join(default_data_dir, "test")
    default_extreme_test_data_dir = os.path.join(default_data_dir, "test")
    default_seasonal_test_data_dir = os.path.join(default_data_dir, "test")

    T_highres_context = {"iid": 10, "ood": 10, "extreme": 20, "seasonal": 70}
    T_highres_target = {"iid": 20, "ood": 20, "extreme": 40, "seasonal": 140}
    T_highres = {key: val_context + val_target
                 for (key, val_context), (_, val_target) in
                 zip(T_highres_context.items(), T_highres_target.items())}
    T_meso = {"iid": 150, "ood": 150, "extreme": 300, "seasonal": 1050}

    def __init__(self,
                 return_mode: str = "default",
                 subset_name: str = "iid",
                 data_dir: Union[Path, str] = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        subset_name:    str
            Name of subset to load from default dir. Must be in ("iid", "ood", "extreme", "seasonal")
        data_dir:   Union[Path, str]
            if `subset_name` is None, use user specified dir
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        super(EarthNet2021TestDataset, self).__init__(
            return_mode=return_mode,
            data_aug_mode=None,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16, )
        '''
        if subset_name == "iid":
            data_dir = self.default_iid_test_data_dir if data_dir is None else data_dir
        elif subset_name == "ood":
            data_dir = self.default_ood_test_data_dir if data_dir is None else data_dir
        elif subset_name == "extreme":
            data_dir = self.default_extreme_test_data_dir if data_dir is None else data_dir
        elif subset_name == "seasonal":
            data_dir = self.default_seasonal_test_data_dir if data_dir is None else data_dir
        else:
            assert subset_name is None  # Use user specified arg data_dir
        '''
        self.subset_name = subset_name
        #self.context_data_dir = self.data_dir.joinpath("context")
        #self.target_data_dir = self.data_dir.joinpath("target")
        #self.context_npz_path_list = sorted(list(self.context_data_dir.glob("**/*.npz")))
        #self.target_npz_path_list = sorted(list(self.target_data_dir.glob("**/*.npz")))
        #if data_dir is None:
        Tdata_dir = self.default_test_dir
        Tsst_data_dir = self.default_sst_test_dir
        Twnd_data_dir = self.default_wnd_test_dir
        Tsla_data_dir = self.default_sla_test_dir

        self.Tdata_dir = Path(Tdata_dir)
        self.Tsst_data_dir = Path(Tsst_data_dir)
        self.Twnd_data_dir = Path(Twnd_data_dir)
        self.Tsla_data_dir = Path(Tsla_data_dir)
        #print ("self.data_dir = ",self.data_dir)
        self.TAnpz_path_list = sorted(list(self.Tdata_dir.glob("SCS_Mercator12_T_daily_*.nc")))
        self.TAsst_path_list = sorted(list(self.Tsst_data_dir.glob("*.nc")))
        self.TAwnd_path_list = sorted(list(self.Twnd_data_dir.glob("*.nc")))
        self.TAsla_path_list = sorted(list(self.Tsla_data_dir.glob("*.nc")))

        self.Tnpz_path_list = sorted(list(self.Tdata_dir.glob("SCS_Mercator12_T_daily_*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.Tsst_path_list = sorted(list(self.Tsst_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.Twnd_path_list = sorted(list(self.Twnd_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        self.Tsla_path_list = sorted(list(self.Tsla_data_dir.glob("*.nc")))[:-in_len*Tint-out_len*1*Tint]
        print ("test number: ",len(self.Tnpz_path_list))

        if len(self.Tnpz_path_list) == len(self.Tsst_path_list) == len(self.Tsla_path_list) == len(self.Twnd_path_list):
           print("所有列表的长度相同")
           print ("self.npz_path_list = ",self.Tnpz_path_list[10])
           print ("self.sst_path_list = ",self.Tsst_path_list[10])
           print ("self.sla_path_list = ",self.Tsla_path_list[10])
           print ("self.wnd_path_list = ",self.Twnd_path_list[10])
        else:
           print("列表的长度不相同，请检查并退出")
           exit()

    def __len__(self) -> int:
        return len(self.Tnpz_path_list)

    def __getitem__(self, idx: int) -> dict:
        #print ("test in ",self.Acontext_npz_path_list[idx])
        #print ("test Fcst ",self.Acontext_npz_path_list[idx+in_len])
        TS = readnc(self.TAnpz_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 

        TS_Pes = readnc(self.TAnpz_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 
        #Laux_data = TS[:,:,:,:4].copy()
        Laux_data = np.random.rand(TS.shape[0],TS.shape[1],TS.shape[2],8).astype(np_dtype)
        Laux_data[:,:,:,0] = readnc_var(self.TAwnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'uwnd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,1] = readnc_var(self.TAwnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'vwnd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,2] = readnc_var(self.TAwnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'wspd',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        Laux_data[:,:,:,3] = readnc_var(self.TAwnd_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'stress_curl',
                                        WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/curl_MAX 
        Laux_data[:,:,:,4] = readnc_SST(self.TAsst_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,
                                        SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 
        Laux_data[:,:,:,5] = readnc_SLA(self.TAsla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'sla',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype)*sla_scale
        Laux_data[:,:,:,6] = readnc_SLA(self.TAsla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'ugos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 
        Laux_data[:,:,:,7] = readnc_SLA(self.TAsla_path_list[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'vgos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 

        SST_Pes = readnc_SST(self.TAsst_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,
                             SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 
        und_Pes = readnc_var(self.TAwnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'uwnd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        vnd_Pes = readnc_var(self.TAwnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'vwnd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        spd_Pes = readnc_var(self.TAwnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'wspd',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/WND_MAX 
        cul_Pes = readnc_var(self.TAwnd_path_list[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,'stress_curl',
                             WND_Blat,WND_Elat,WND_Blon,WND_Elon,Tdvar=False)[:,:,:,0].astype(np_dtype)/curl_MAX 

        for i in range(in_len+out_len):
              TS[i,:,:,:]=TS[i,:,:,:]-TS_Pes[0,:,:,:]
              Laux_data[i,:,:,0]=Laux_data[i,:,:,0]-und_Pes[0,:,:]
              Laux_data[i,:,:,1]=Laux_data[i,:,:,1]-vnd_Pes[0,:,:]
              Laux_data[i,:,:,2]=Laux_data[i,:,:,2]-spd_Pes[0,:,:]
              Laux_data[i,:,:,3]=Laux_data[i,:,:,3]-cul_Pes[0,:,:]
              Laux_data[i,:,:,4]=Laux_data[i,:,:,4]-SST_Pes[0,:,:]
              #plot_var(TS[i,:,:,0],'TS'+str(i))

        return {"highresdynamic": TS,"auxiliary": Laux_data,}
        '''
        #print (self.context_npz_path_list[idx])
        #print (self.target_npz_path_list[idx])
        context_data_npz = np.load(self.context_npz_path_list[idx])
        target_data_npz = np.load(self.target_npz_path_list[idx])

        context_highresdynamic = context_data_npz["highresdynamic"].astype(self.np_dtype)
        highresstatic = context_data_npz["highresstatic"].astype(self.np_dtype)
        mesodynamic = context_data_npz["mesodynamic"].astype(self.np_dtype)
        mesostatic = context_data_npz["mesostatic"].astype(self.np_dtype)
        target_highresdynamic = target_data_npz["highresdynamic"].astype(self.np_dtype)
        highresdynamic = np.concatenate([context_highresdynamic,
                                         target_highresdynamic],
                                        axis=-1)

        return self.process_raw_data_from_npz(highresdynamic, highresstatic,
                                              mesodynamic, mesostatic)
        '''

class EarthNet2021LightningDataModule(LightningDataModule):

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 train_data_dir: Union[Path, str] = None,
                 test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
                 test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
                 val_ratio: float = 0.1,
                 train_val_split_seed: int = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False,
                 # datamodule_only
                 batch_size=1,
                 num_workers=8, ):
        super(EarthNet2021LightningDataModule, self).__init__()
        self.return_mode = return_mode
        self.data_aug_mode = data_aug_mode
        self.data_aug_cfg = data_aug_cfg
        if train_data_dir is None:
            train_data_dir = EarthNet2021TrainDataset.default_train_dir
        self.train_data_dir = train_data_dir

        if test_data_dir is None:
            test_data_dir = EarthNet2021TestDataset.default_test_dir
        self.test_data_dir = test_data_dir

        if test_subset_name is None:
            if not isinstance(test_data_dir, Sequence):
                self.test_data_dir_list = [test_data_dir, ]
            else:
                self.test_data_dir_list = list(test_data_dir)
            self.test_subset_name_list = [None, ] * len(self.test_data_dir_list)
        else:
            if isinstance(test_subset_name, str):
                self.test_subset_name_list = [test_subset_name, ]
            elif isinstance(test_subset_name, Sequence):
                self.test_subset_name_list = list(test_subset_name)
            else:
                raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
            '''
            self.test_data_dir_list = []
            for test_subset_name in self.test_subset_name_list:
                if test_subset_name == "iid":
                    test_data_dir = EarthNet2021TestDataset.default_iid_test_data_dir
                elif test_subset_name == "ood":
                    test_data_dir = EarthNet2021TestDataset.default_ood_test_data_dir
                elif test_subset_name == "extreme":
                    test_data_dir = EarthNet2021TestDataset.default_extreme_test_data_dir
                elif test_subset_name == "seasonal":
                    test_data_dir = EarthNet2021TestDataset.default_seasonal_test_data_dir
                else:
                    raise ValueError(f"Invalid test_subset_name {test_subset_name}")
                self.test_data_dir_list.append(test_data_dir)
            '''

        self.val_ratio = val_ratio
        self.train_val_split_seed = train_val_split_seed

        self.layout = layout
        self.static_layout = static_layout
        self.highresstatic_expand_t = highresstatic_expand_t
        self.mesostatic_expand_t = mesostatic_expand_t
        self.meso_crop = meso_crop
        self.fp16 = fp16
        # datamodule_only
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        assert os.path.exists(self.train_data_dir), "EarthNet2021 training set not found!"
        for test_data_dir in self.test_data_dir_list:
            assert os.path.exists(test_data_dir), f"EarthNet2021 test set at {test_data_dir} not found!"

    def setup(self, stage = None):
        if stage in (None, "fit"):
            train_val_data = EarthNet2021TrainDataset(
                return_mode=self.return_mode,
                data_aug_mode=self.data_aug_mode,
                data_aug_cfg=self.data_aug_cfg,
                data_dir=self.train_data_dir,
                layout=self.layout,
                static_layout=self.static_layout,
                highresstatic_expand_t=self.highresstatic_expand_t,
                mesostatic_expand_t=self.mesostatic_expand_t,
                meso_crop=self.meso_crop,
                fp16=self.fp16)
            val_size = int(self.val_ratio * len(train_val_data))
            train_size = len(train_val_data) - val_size

            if self.train_val_split_seed is not None:
                rnd_generator_dict = dict(generator=torch.Generator().manual_seed(self.train_val_split_seed))
            else:
                rnd_generator_dict = {}
            #self.earthnet_train, self.earthnet_val = random_split(
            #    train_val_data, [train_size, val_size],
            #    **rnd_generator_dict)
            print ("No random_split")
            print (train_val_data.size())
            self.earthnet_train, self.earthnet_val = train_val_data[:train_size,:,:,:],train_val_data[train_size+1:,:,:,:]

        if stage in (None, "test"):
            self.earthnet_test_list = [
                EarthNet2021TestDataset(
                    return_mode=self.return_mode,
                    subset_name=test_subset_name,
                    data_dir=test_data_dir,
                    layout=self.layout,
                    static_layout=self.static_layout,
                    highresstatic_expand_t=self.highresstatic_expand_t,
                    mesostatic_expand_t=self.mesostatic_expand_t,
                    meso_crop=self.meso_crop,
                    fp16=self.fp16)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

        if stage in (None, "predict"):
            self.earthnet_predict_list = [
                EarthNet2021TestDataset(
                    return_mode=self.return_mode,
                    subset_name=test_subset_name,
                    data_dir=test_data_dir,
                    layout=self.layout,
                    static_layout=self.static_layout,
                    highresstatic_expand_t=self.highresstatic_expand_t,
                    mesostatic_expand_t=self.mesostatic_expand_t,
                    meso_crop=self.meso_crop,
                    fp16=self.fp16)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

    @property
    def num_train_samples(self):
        return len(self.earthnet_train)

    @property
    def num_val_samples(self):
        return len(self.earthnet_val)

    @property
    def num_test_samples(self):
        if len(self.earthnet_test_list) == 1:
            return len(self.earthnet_test_list[0])
        else:
            return [len(earthnet_test) for earthnet_test in self.earthnet_test_list]

    @property
    def num_predict_samples(self):
        if len(self.earthnet_predict_list) == 1:
            return len(self.earthnet_predict_list[0])
        else:
            return [len(earthnet_predict) for earthnet_predict in self.earthnet_predict_list]

    def train_dataloader(self):
        return DataLoader(self.earthnet_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.earthnet_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if len(self.earthnet_test_list) == 1:
            return DataLoader(self.earthnet_test_list[0], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            return [DataLoader(earthnet_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                    for earthnet_test in self.earthnet_test_list]

    def predict_dataloader(self):
        if len(self.earthnet_predict_list) == 1:
            return DataLoader(self.earthnet_predict_list[0], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            return [
                DataLoader(earthnet_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                for earthnet_predict in self.earthnet_predict_list]

def get_EarthNet2021_dataloaders(
        dataloader_return_mode: str = "default",
        data_aug_mode: str = None,
        data_aug_cfg: Dict = None,
        train_data_dir: Union[Path, str] = None,
        test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
        test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
        val_ratio: float = 0.1,
        train_val_split_seed: int = None,
        layout: str = None,
        static_layout: str = None,
        highresstatic_expand_t: bool = False,
        mesostatic_expand_t: bool = False,
        meso_crop: Union[str, Sequence[Sequence[int]]] = None,
        fp16: bool = False,
        batch_size=1,
        num_workers=4, ):

    if test_subset_name is None:
        if not isinstance(test_data_dir, Sequence):
            test_data_dir_list = [test_data_dir, ]
        else:
            test_data_dir_list = list(test_data_dir)
        test_subset_name_list = [None, ] * len(test_data_dir_list)
    else:
        if isinstance(test_subset_name, str):
            test_subset_name_list = [test_subset_name, ]
        elif isinstance(test_subset_name, Sequence):
            test_subset_name_list = list(test_subset_name)
        else:
            raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
        test_data_dir_list = []
        for test_subset_name in test_subset_name_list:
            if test_subset_name == "iid":
                test_data_dir = EarthNet2021TestDataset.default_iid_test_data_dir
            elif test_subset_name == "ood":
                test_data_dir = EarthNet2021TestDataset.default_ood_test_data_dir
            elif test_subset_name == "extreme":
                test_data_dir = EarthNet2021TestDataset.default_extreme_test_data_dir
            elif test_subset_name == "seasonal":
                test_data_dir = EarthNet2021TestDataset.default_seasonal_test_data_dir
            else:
                raise ValueError(f"Invalid test_subset_name {test_subset_name}")
            test_data_dir_list.append(test_data_dir)

    train_val_data = EarthNet2021TrainDataset(
        return_mode=dataloader_return_mode,
        data_aug_mode=data_aug_mode,
        data_aug_cfg=data_aug_cfg,
        data_dir=train_data_dir,
        layout=layout,
        static_layout=static_layout,
        highresstatic_expand_t=highresstatic_expand_t,
        mesostatic_expand_t=mesostatic_expand_t,
        meso_crop=meso_crop,
        fp16=fp16)
    val_size = int(val_ratio * len(train_val_data))
    train_size = len(train_val_data) - val_size

    if train_val_split_seed is not None:
        rnd_generator_dict = dict(generator=torch.Generator().manual_seed(train_val_split_seed))
    else:
        rnd_generator_dict = {}
    earthnet_train, earthnet_val = random_split(
        train_val_data, [train_size, val_size],
        **rnd_generator_dict)
    print (earthnet_train)
    #print ("No random_split")
    #print (len(train_val_data))
    #print (type(train_val_data))
    #self.earthnet_train, self.earthnet_val = train_val_data[:train_size],train_val_data[train_size+1:]

    earthnet_test_list = [
        EarthNet2021TestDataset(
            return_mode=dataloader_return_mode,
            subset_name=test_subset_name,
            data_dir=test_data_dir,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16)
        for test_subset_name, test_data_dir in
        zip(test_subset_name_list, test_data_dir_list)]

    num_test_samples = [len(earthnet_test) for earthnet_test in earthnet_test_list]
    test_dataloader = [DataLoader(earthnet_test, batch_size=batch_size, shuffle=False, num_workers=1)
                       for earthnet_test in earthnet_test_list]
    
    return {
        "train_dataloader": DataLoader(earthnet_train, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "val_dataloader": DataLoader(earthnet_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_dataloader": test_dataloader,
        "num_train_samples": len(earthnet_train),
        "num_val_samples": len(earthnet_val),
        "num_test_samples": num_test_samples,
    }
