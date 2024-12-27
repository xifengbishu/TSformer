#coding:utf-8
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import warnings
from typing import Sequence
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import structural_similarity_index_measure
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from lightning.pytorch.strategies import FSDPStrategy
#from pytorch_lightning.strategies import FSDPStrategy
from omegaconf import OmegaConf
import os
from pathlib import Path
import argparse
from einops import rearrange

from src.earthformer.config import cfg
from src.earthformer.utils.optim import SequentialLR, warmup_lambda
from src.earthformer.utils.utils import get_parameter_names
from src.earthformer.utils.layout import layout_to_in_out_slice
#from src.earthformer.visualization.nbody import save_example_vis_results
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
#from src.earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from src.earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel

#from earthformer.datasets.earthnet.SLA12_dataloader import EarthNet2021LightningDataModule, get_EarthNet2021_dataloaders
#from earthformer.datasets.earthnet.BAK_DataloaderTS import EarthNet2021LightningDataModule, get_EarthNet2021_dataloaders
from LoaderTS import EarthNet2021LightningDataModule, get_EarthNet2021_dataloaders
#from earthformer.datasets.earthnet.DDDataloaderTS import EarthNet2021LightningDataModule, get_EarthNet2021_dataloaders
#from earthformer.datasets.earthnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
from src.earthformer.datasets.earthnet.visualization import vis_earthnet_seq
#from src.earthformer.utils.apex_ddp import ApexDDPStrategy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
import deepspeed

import netCDF4 as nc
from datetime import datetime
from TTWGS_nc_lib import plot_var,readnc_SLA,readnc_SST,readnc,write_to_nc_4dvar 

print ("------------------------------------------------")
torch.cuda.memory_summary()
torch.cuda.empty_cache()
'''
# 设置 CUDA_VISIBLE_DEVICES 环境变量，使得脚本可以使用8个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Set the precision for float32 matrix multiplications
torch.set_float32_matmul_precision('medium')  # or 'high'
Strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
'''
#print (os.system('clear'))
print ("                                                ")
print ("------------ Python Lib load OK ----------------")
print ("                                                ")
print ("torch verison = ", torch.__version__)
print ("pytorch_lightning verison = ", pl.__version__)
print ("                                                ")
print ("------------------------------------------------")
print ("                                                ")
print (torch.cuda.is_available())
print (torch.cuda.device_count())
print ()


YMD = "mean_WGSYMD"
YMD = "WGSYMD"

Anum  = 30
dxdy=12
SST_dxdy = 20
SLA_dxdy = 4
sla_scale = 4.0

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

lato=5
lono=105
nlat=240
nlon=240

lato=10
lono=110
nlat=180
nlon=180

lato=20
lono=120
nlat=120
nlon=120

lato=0
lono=100
nlat=360
nlon=360

lato=2
lono=105
nlat=312
nlon=240

lato=-80
lono=0
nlat=1020
nlon=1020

lato=0
lono=100
nlat=360
nlon=360

lato=-80
lono=0
nlat=WGS_nlat
nlon=WGS_nlon

Pauxiliary_channels = WGS_Paux
Tint = WGS_Tint
in_len = WGS_in_len
out_len = WGS_out_len
in_channels = WGS_in_channels
Cnum = int(Anum/out_len)
print (Cnum)
ot_channels = 1
#Slev=[0,4,7,10,12,13,14,15,16,17,20,21,22,23,24,25]
Slev=[0,4,7,10,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
#Slev=[0,2,3,4,5,6,7,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
#Slev=[0,7,12,14,16]
# 0.5 9.5 29, 47 
#Slev=[0,7,17]

#Slev=[0,7,12,14,16,17,20,21,23,24,25,26]
#Slev=[0]
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

# ------------
SST_Blat=(90+lato)*SST_dxdy
SST_Elat=int(SST_Blat+(nlat*SST_dxdy/12))
SST_Blon=(180+lono)*SST_dxdy
SST_Elon=int(SST_Blon+(nlon*SST_dxdy/12))

SLA_Blat=(90+lato)*SLA_dxdy
SLA_Elat=int(SLA_Blat+(nlat*SLA_dxdy/12))
SLA_Blon=(0+lono)*SLA_dxdy
#SLA_Blon=(180+lono)*SLA_dxdy
SLA_Elon=int(SLA_Blon+(nlon*SLA_dxdy/12))

# ------------
SST_Blat=(lato+1)*SST_dxdy
SST_Elat=int(SST_Blat+(nlat*SST_dxdy/12))
SST_Blon=(lono-99)*SST_dxdy
SST_Elon=int(SST_Blon+(nlon*SST_dxdy/12))

SLA_Blat=(lato+1)*SLA_dxdy
SLA_Elat=int(SLA_Blat+(nlat*SLA_dxdy/12))
SLA_Blon=(lono-99)*SLA_dxdy
#SLA_Blon=(180+lono)*SLA_dxdy
SLA_Elon=int(SLA_Blon+(nlon*SLA_dxdy/12))

# ------------------

dx = 1.0/dxdy
RBlat=-80+Blat*dx
RElat=-80+Elat*dx
RBlon= -180+Blon*dx
RElon= -180+Elon*dx

#TS_data_dir = "/ooi_data/WGS_DATA/Mercatorglorys12v1/Mercator_glorys1v12_2019/test"
#SST_data_dir = "/ooi_data/WGS_DATA/SST/OSTIA/OSTIA_2019/test"
#SLA_data_dir = "/home/wind/2Project3-MEIT/NRT_download/AVISO/DT_DATA/2019/test"

#TS_data_dir = "/home/wang/Data1/Mercator/Mercator_glorys12v1/test"
#SST_data_dir = "/home/wang/Data/SST/OSTIA/test"
#SLA_data_dir = "/home/wang/Data1/AVISO/test"

if str(in_channels) == '1':
   TS_data_dir = "/home/user/145_SRRAM_SCS/DATA/Mercatorglorys12v1_SCS/thetao/test"
else :
   TS_data_dir = "/home/user/145_SRRAM_SCS/DATA/Mercatorglorys12v1_SCS/so/test"

SST_data_dir = "/home/user/145_SRRAM_SCS/DATA/OSTIA/DT_DATA/test"
SLA_data_dir = "/home/user/145_SRRAM_SCS/DATA/AVISO/test"

#TS_data_dir = "/gpfs02/home/user/145-SRRAM-SCS/3DTS-Project3/Unet-TSformer/DATA/Mercator_glorys12v1"
#SST_data_dir = "/gpfs02/home/user/145-SRRAM-SCS/3DTS-Project3/Unet-TSformer/DATA/OSTIA"
#SLA_data_dir = "/gpfs02/home/user/145-SRRAM-SCS/3DTS-Project3/Unet-TSformer/DATA/AVISO"


_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
#pretrained_checkpoints_dir = os.path.join(_curr_dir, "Pretrained_checkpoints")
#pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
print ("cuur_dir ",_curr_dir)
#print ("pretrained_checkpoints_dir ",pretrained_checkpoints_dir)
pytorch_state_dict_name = "Unet_TSformer.pt"
np_dtype = np.float32 


clim = np.random.rand(len(Slev),int(Elat-Blat),int(Elon-Blon)).astype(np_dtype)
ori_data  = nc.Dataset('mercatorglorys12v1_gl12_mean_1993-2012.nc')     # 读取nc文件
ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
depth = ori_variables['depth'][Slev]

if str(in_channels) == '1':
  clim = ori_variables['thetao'][0,Slev,ABlat:AElat,ABlon:AElon]
else :
  clim = ori_variables['so'][0,Slev,ABlat:AElat,ABlon:AElon]
ori_data.close()

clim[clim<-100.] = -9999.
clim[clim> 100.] = -9999.
#clim=np.nan_to_num(clim)
clim = clim.filled(-9999.)
print ("clim.shape = ",clim.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add this function to initialize DeepSpeed
def initialize_deepspeed(pl_module, args):
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True
        },
        "fp16": {
            "enabled": True
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pl_module.torch_nn_module,
        model_parameters=pl_module.torch_nn_module.parameters(),
        config_params=deepspeed_config
    )
    return model_engine, optimizer

class CuboidEarthNet2021PLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidEarthNet2021PLModule, self).__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerAuxModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
            # different from CuboidTransformerModel, no arg `dec_use_first_self_attn=False`
            #data_channels=model_cfg["data_channels"],
            auxiliary_channels=model_cfg["auxiliary_channels"],
            unet_dec_cross_mode=model_cfg["unet_dec_cross_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.channels = model_cfg["data_channels"]
        self.auxiliary_channels = model_cfg["auxiliary_channels"]
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        test_subset_name = oc.dataset.test_subset_name
        if isinstance(test_subset_name, Sequence):
            test_subset_name = list(test_subset_name)
        else:
            test_subset_name = [test_subset_name, ]
        self.test_subset_name = test_subset_name

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        #self.valid_ens = EarthNet2021ScoreUpdateWithoutCompute(layout=self.layout, eps=1E-4)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        #self.test_ens = EarthNet2021ScoreUpdateWithoutCompute(layout=self.layout, eps=1E-4)

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 10
        cfg.out_len = 20
        cfg.img_height = 128
        cfg.img_width = 128
        cfg.layout = "NTHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        layout_cfg = cls.get_layout_config()
        cfg.data_channels = 4
        cfg.input_shape = (layout_cfg.in_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.target_shape = (layout_cfg.out_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True

        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        # different from CuboidTransformerModel, no arg `dec_use_first_self_attn=False`
        cfg.auxiliary_channels = 4  # 5 from mesodynamic, 1 from highresstatic, 1 from mesostatic
        cfg.unet_dec_cross_mode = "up"
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.return_mode = "default"
        cfg.data_aug_mode = None
        cfg.data_aug_cfg = None
        cfg.test_subset_name = ("iid", "ood")
        layout_cfg = cls.get_layout_config()
        cfg.in_len = layout_cfg.in_len
        cfg.out_len = layout_cfg.out_len
        cfg.layout = "THWC"
        cfg.static_layout = "CHW"
        cfg.val_ratio = 0.1
        cfg.train_val_split_seed = None
        cfg.highresstatic_expand_t = False
        cfg.mesostatic_expand_t = False
        cfg.meso_crop = None
        cfg.fp16 = False        
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = None
        cfg.total_batch_size = 32
        cfg.micro_batch_size = 8

        cfg.method = "adamw"
        cfg.lr = 1E-3
        cfg.wd = 1E-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        # scheduler
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        cfg.min_lr_ratio = 0.1
        cfg.warmup_min_lr_ratio = 0.1
        # early stopping
        cfg.early_stop = False
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "EarthNet2021"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        #cfg.track_grad_norm = -1
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        cfg.precision = 32
        return cfg

    @staticmethod
    def get_vis_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0, ]
        cfg.val_example_data_idx_list = [0, ]
        cfg.test_example_data_idx_list = [0, ]
        cfg.eval_example_only = False
        cfg.ncols = 10
        cfg.dpi = 300
        cfg.figsize = None
        cfg.font_size = 10
        cfg.y_label_rotation = 0
        cfg.y_label_offset = (-0.05, 0.4)
        return cfg

    '''
    def configure_optimizers(self):
        return FusedAdam(self.parameters())

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        return DeepSpeedCPUAdam(self.parameters())
    '''
    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
            #optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters,
            #                              lr=self.oc.optim.lr,
            #                              weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            #track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            #strategy="ddp",
            #strategy="deepspeed",
            #strategy="fsdp",
            #strategy="deepspeed_stage_1",
            strategy="deepspeed_stage_2",
            #strategy = DDPStrategy(find_unused_parameters=True),
            #strategy = DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
            #strategy = "deepspeed_stage_2_offload",
            #strategy = "deepspeed_stage_3",
            #strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_earthnet2021_datamodule(
            dataset_cfg,
            micro_batch_size: int = 1,
            num_workers: int = 8):
        dm = EarthNet2021LightningDataModule(
            return_mode=dataset_cfg["return_mode"],
            data_aug_mode=dataset_cfg["data_aug_mode"],
            data_aug_cfg=dataset_cfg["data_aug_cfg"],
            val_ratio=dataset_cfg["val_ratio"],
            train_val_split_seed=dataset_cfg["train_val_split_seed"],
            layout=dataset_cfg["layout"],
            static_layout=dataset_cfg["static_layout"],
            highresstatic_expand_t=dataset_cfg["highresstatic_expand_t"],
            mesostatic_expand_t=dataset_cfg["mesostatic_expand_t"],
            meso_crop=dataset_cfg["meso_crop"],
            fp16=dataset_cfg["fp16"],
            # datamodule_only
            batch_size=micro_batch_size,
            num_workers=num_workers, )
        return dm

    @staticmethod
    def get_earthnet2021_dataloaders(
            dataset_cfg,
            micro_batch_size: int = 1,
            num_workers: int = 8):
        dataloader_dict = get_EarthNet2021_dataloaders(
            dataloader_return_mode=dataset_cfg["return_mode"],
            data_aug_mode=dataset_cfg["data_aug_mode"],
            data_aug_cfg=dataset_cfg["data_aug_cfg"],
            test_subset_name=dataset_cfg["test_subset_name"],
            val_ratio=dataset_cfg["val_ratio"],
            train_val_split_seed=dataset_cfg["train_val_split_seed"],
            layout=dataset_cfg["layout"],
            static_layout=dataset_cfg["static_layout"],
            highresstatic_expand_t=dataset_cfg["highresstatic_expand_t"],
            mesostatic_expand_t=dataset_cfg["mesostatic_expand_t"],
            meso_crop=dataset_cfg["meso_crop"],
            fp16=dataset_cfg["fp16"],
            batch_size=micro_batch_size,
            num_workers=num_workers, )
        return dataloader_dict

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def forward(self, batch):
        highresdynamic = batch["highresdynamic"]
        auxiliary      = batch["auxiliary"]
        seq = highresdynamic[..., :self.channels]
        # mask from dataloader: 1 for mask and 0 for non-masked
        mask = highresdynamic[..., self.channels-1: self.channels][self.out_slice]
        mask = mask-mask

        in_seq = seq[self.in_slice]
        target_seq = seq[self.out_slice]
        #print (self.auxiliary_channels)
        #print (self.channels)
        #Laux_data = highresdynamic[..., :self.auxiliary_channels].clone()
        #Laux_data[:,in_seq.size(1):,:,:,:] = Laux_data[:,in_seq.size(1)-1:in_seq.size(1),:,:,:].repeat(1, target_seq.size(1),1, 1, 1)
        if self.auxiliary_channels == 1 and Pauxiliary_channels == 0 :
              #print ("SST")
              Laux_data = auxiliary[..., :self.auxiliary_channels].clone()
        elif self.auxiliary_channels == 2 and Pauxiliary_channels == 0 :
              #print ("SST+SLA")
              Laux_data = auxiliary[..., :self.auxiliary_channels].clone()
        elif self.auxiliary_channels == 4 and Pauxiliary_channels == 0 :
              #print ("SST+SLA+Ugos+Vgos")
              Laux_data = auxiliary[..., :self.auxiliary_channels].clone()
        else :
              #print ("persis")
              Laux_data = highresdynamic[..., :1].clone()
              Laux_data[:,in_seq.size(1):,:,:,:] = Laux_data[:,in_seq.size(1)-1:in_seq.size(1),:,:,:].repeat(1, target_seq.size(1),1, 1, 1)
        #print ("Laux_data.size",Laux_data.size())
        #print ("in_seq.size",in_seq.size())
        #print ("targer_seq.size",target_seq.size())

        pred_seq = self.torch_nn_module(in_seq, Laux_data[self.in_slice], Laux_data[self.out_slice])
        #print ("pred_seq 1 ",pred_seq[0,:,0,0,0])
        #print ("in_seq ",in_seq[0,:,0,0,0])
        #print ("in_seq ",in_seq[0,in_seq.size(1)-1,0,0,0])
        #pred_seq = pred_seq + Laux_data[:,in_seq.size(1):,:,:,:]
        #print ("pred_seq.size",pred_seq.size()) #pred_seq.size torch.Size([2, 5, 120, 120, 26])
        #for i in range(in_seq.size(1)):
        #    pred_seq[:,i,:,:,:] = pred_seq[:,i,:,:,:] + in_seq[:,in_seq.size(1)-1,:,:,:]

        loss_mse = F.mse_loss(pred_seq * (1 - mask), target_seq * (1 - mask))
        loss_mae = self.valid_mae(pred_seq * (1 - mask), target_seq * (1 - mask))
        #ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        #loss_ssim = ssim(pred_seq * (1 - mask), target_seq * (1 - mask))
        #loss_ssim = structural_similarity_index_measure(pred_seq * (1 - mask), target_seq * (1 - mask))
        #loss_ssim = structural_similarity_index_measure(pred_seq[:,:,:,:,0],target_seq[:,:,:,:,0])
        #print ("loss_mse",loss_mse)
        #print ("loss_mae",loss_mae)
        #print ("--- loss mse mae ssim ",loss_mse[:],loss_mae[:],loss_ssim[:],"---")
        #print ("-------------------")
        loss = loss_mae + loss_mse
        #loss = (1.0 -loss_ssim) + loss_mae + loss_mse
        #print ("loss",loss)
        #loss = F.mse_loss(pred_seq,target_seq)
        #loss = F.mse_loss(pred_seq * (1 - mask), target_seq * (1 - mask))
        #exit()
        return pred_seq, loss, in_seq, target_seq, mask

    def training_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq, mask = self(batch)
        micro_batch_size = in_seq.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=in_seq.detach().float().cpu().numpy(),
                target_seq=target_seq.detach().float().cpu().numpy(),
                pred_seq=pred_seq.detach().float().cpu().numpy(),
                mode="train", )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[list(batch.keys())[0]].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, mask = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="val", )
            #if self.precision == 16:
            #    pred_seq = pred_seq.float()
            self.valid_mse(pred_seq * (1 - mask), target_seq * (1 - mask))
            self.valid_mae(pred_seq * (1 - mask), target_seq * (1 - mask))
            #self.valid_ens(pred_seq, target_seq, mask)
        return None

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        #valid_ens_dict = self.valid_ens.compute()
        #valid_loss = -valid_ens_dict["EarthNetScore"]
        valid_loss = valid_mae+valid_mse

        self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
        #self.log('valid_MAD_epoch', valid_ens_dict["MAD"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log('valid_OLS_epoch', valid_ens_dict["OLS"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log('valid_EMD_epoch', valid_ens_dict["EMD"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log('valid_SSIM_epoch', valid_ens_dict["SSIM"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log('valid_EarthNetScore_epoch', valid_ens_dict["EarthNetScore"], prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        #self.valid_ens.reset()

    @property
    def test_epoch_count(self):
        if not hasattr(self, "_test_epoch_count"):
            self.reset_test_epoch_count()
        return self._test_epoch_count

    def increase_test_epoch_count(self, val=1):
        if not hasattr(self, "_test_epoch_count"):
            self.reset_test_epoch_count()
        self._test_epoch_count += val

    def reset_test_epoch_count(self):
        self._test_epoch_count = 0

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[list(batch.keys())[0]].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, mask = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="test",
                    prefix=f"{self.test_subset_name[self.test_epoch_count]}_")
            #if self.precision == 16:
            #    pred_seq = pred_seq.float()
            self.test_mse(pred_seq * (1 - mask), target_seq * (1 - mask))
            self.test_mae(pred_seq * (1 - mask), target_seq * (1 - mask))
            #self.test_ens(pred_seq, target_seq, mask)
        return None

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        #test_ens_dict = self.test_ens.compute()

        prefix = self.test_subset_name[self.test_epoch_count]
        self.log(f'{prefix}_test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f'{prefix}_test_MAD_epoch', test_ens_dict["MAD"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f'{prefix}_test_OLS_epoch', test_ens_dict["OLS"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f'{prefix}_test_EMD_epoch', test_ens_dict["EMD"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f'{prefix}_test_SSIM_epoch', test_ens_dict["SSIM"], prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f'{prefix}_test_EarthNetScore_epoch', test_ens_dict["EarthNetScore"], prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        #self.test_ens.reset()

        self.increase_test_epoch_count()

    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: np.ndarray,
            mode: str = "train",
            prefix: str = ""):
        r"""
        Parameters
        ----------
        data_idx
        context_seq, target_seq, pred_seq:   np.ndarray
            layout should not include batch
        mode:   str
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            #if data_idx in example_data_idx_list:
            #    for variable in ["rgb", "ndvi"]:
            #        save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_{variable}_data_{data_idx}.png"
            #        vis_earthnet_seq(
            #            context_np=context_seq,
            #            target_np=target_seq,
            #            pred_np=pred_seq,
            #            ncols=self.oc.vis.ncols,
            #            layout=self.layout,
            #            variable=variable,
            #            vegetation_mask=None,
            #            cloud_mask=True,
            #            save_path=os.path.join(self.example_save_dir, save_name),
            #            dpi=self.oc.vis.dpi,
            #            figsize=self.oc.vis.figsize,
            #            font_size=self.oc.vis.font_size,
            #            y_label_rotation=self.oc.vis.y_label_rotation,
            #            y_label_offset=self.oc.vis.y_label_offset, )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_earthnet2021', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on EarthNet2021.')
    return parser



def get_args():
    parser = argparse.ArgumentParser(description='TSformer')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser.add_argument('--save', default='tmp_earthnet2021', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args





if not os.path.exists('./Forecast_Result_'+str(YMD)+'/'):
    os.mkdir('./Forecast_Result_'+str(YMD)+'/')

if not os.path.exists('./Verify/'):
    os.mkdir('./Verify/')

# Get the list of all files in the folder
file_list = os.listdir('./Verify')

# Iterate through the list and delete each file
for file_name in file_list:
    file_path = os.path.join('./Verify', file_name)
    print (file_path)
    os.remove(file_path)
    #if os.path.isfile(file_path):
    #    os.remove(file_path)
def Finput(idx,TS_Flist,SST_Flist,SLA_Flist,Cycle=False):
	print ("SST forcast begin:",SST_Flist[idx+in_len*Tint])
	print ("SLA forcast begin:",SLA_Flist[idx+in_len*Tint])
	if not Cycle:
		print ("TS forcast begin:",TS_Flist[idx+in_len*Tint])
	print("*************************************************************")
	if not Cycle:
		for file in TS_Flist[idx:idx+in_len*Tint:Tint]:
			print("input fle = ",file.name)

		# Source and destination file paths
		src_file = TS_Flist[idx+(in_len-1)*Tint]
		dest_dir = "./Verify/"
		# Create a hard link
		#os.link(src_file, dest_file)
		# Create a symbolic link to the file in the destination directory
		#os.symlink(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
		if not os.path.islink(os.path.join(dest_dir, os.path.basename(src_file))):
			# Create a symbolic link to the file in the destination directory
			os.symlink(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
		else:
			print(f"The symbolic link '{os.path.join(dest_dir, os.path.basename(src_file))}' already exists.")

		print("*************************************************************")
		for file in TS_Flist[idx+in_len*Tint:idx+in_len*Tint+out_len*Tint:Tint]:
			print("output fle = ",file.name)
			src_file = TS_data_dir+'/'+str(file.name)
			#os.link(src_file, dest_file)
			#os.symlink(src_file, os.path.join(dest_dir, os.path.basename(src_file)))

			# Check if the symbolic link already exists
			if not os.path.islink(os.path.join(dest_dir, os.path.basename(src_file))):
				# Create a symbolic link to the file in the destination directory
				os.symlink(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
			else:
				print(f"The symbolic link '{os.path.join(dest_dir, os.path.basename(src_file))}' already exists.")

		TS = readnc(TS_Flist[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 
		#print ("TS shape",TS.shape)
		print ("TS_Pes:",TS_Flist[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1])
		TS_Pes = readnc(TS_Flist[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 
		#TS_Pes = readnc(TS_Flist[idx+in_len*Tint+Tint-1:idx+in_len*Tint+Tint-0],Blat,Elat,Blon,Elon,Slev,in_channels,clim).astype(np_dtype) 

	SST_Pes = readnc_SST(SST_Flist[idx+in_len*Tint-Tint:idx+in_len*Tint-Tint+1],Blat,Elat,Blon,Elon,
                             SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 

	#Laux_data = TS[:,:,:,:4].copy()
	Laux_data = np.random.rand(in_len+out_len,int(Elat-Blat),int(Elon-Blon),4).astype(np_dtype)
	#Laux_data = np.zeros((in_len+out_len,nlat,nlon,4)).astype(np_dtype)
	#print ("Laux_data shape",Laux_data.shape)
        #TS = np.zeros(pred.shape[0]*pred.shape[1],pred.shape[2],pred.shape[3],pred.shape[4],pred.shape[5]).astype(np_dtype)
	Laux_data[:,:,:,0] = readnc_SST(SST_Flist[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,
                                        SST_Blat,SST_Elat,SST_Blon,SST_Elon,in_channels,clim)[:,:,:,0].astype(np_dtype) 
	Laux_data[:,:,:,1] = readnc_SLA(SLA_Flist[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'sla',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype)*sla_scale
	Laux_data[:,:,:,2] = readnc_SLA(SLA_Flist[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'ugos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 
	Laux_data[:,:,:,3] = readnc_SLA(SLA_Flist[idx:idx+in_len*Tint+out_len*Tint:Tint],Blat,Elat,Blon,Elon,'vgos',
                                        SLA_Blat,SLA_Elat,SLA_Blon,SLA_Elon,in_channels)[:,:,:,0].astype(np_dtype) 

	for i in range(in_len+out_len):
		if not Cycle:
			TS[i,:,:,:]=TS[i,:,:,:]-TS_Pes[0,:,:,:]
		Laux_data[i,:,:,0]=Laux_data[i,:,:,0]-SST_Pes[0,:,:]
	print (Laux_data[in_len-2:in_len+1,0,0,0])

	if not Cycle:
		print (TS[in_len-2:in_len+1,0,0,0])
		return {"highresdynamic": torch.tensor(np.expand_dims(TS, axis=0)),"auxiliary": torch.tensor(np.expand_dims(Laux_data, axis=0)),"TS_Pes": torch.tensor(np.expand_dims(TS_Pes,axis=0))}
	else :
		return {"auxiliary": torch.tensor(np.expand_dims(Laux_data, axis=0))}

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def TS_input(TS,Laux_data,TS_Pes):
	#TS_data = np.zeros((1,in_len+out_len,nlat,nlon,len(Slev))).astype(np_dtype)
	TS_data = np.random.rand(1,in_len+out_len,nlat,nlon,len(Slev)).astype(np_dtype)
	TS_data[:,:in_len,:,:,:] = TS
	return {"highresdynamic": torch.tensor(TS_data),"auxiliary": torch.tensor(Laux_data),"TS_Pes": torch.tensor(TS_Pes)}



def TS_forecast(input_data,pl_module):
	with torch.no_grad():
		pred_seq, loss,iput_seq, tget_seq, mask = pl_module(input_data)
		pred = pred_seq.detach().cpu().numpy()
		tget = tget_seq.detach().cpu().numpy()
		iput = iput_seq.detach().cpu().numpy()

		#pess = input_data["TS_Pes"].detach().cpu().numpy()
		#for i in range(out_len):
		#	pred[:,i,:,:,:]=pred[:,i,:,:,:]+pess[:,0,:,:,:]
		#	tget[:,i,:,:,:]=tget[:,i,:,:,:]+pess[:,0,:,:,:]
		#for i in range(in_len):
		#	iput[:,i,:,:,:]=iput[:,i,:,:,:]+pess[:,0,:,:,:]
		print (pred.shape)
	return pred,tget,iput

# Add this function to initialize DeepSpeed
def initialize_deepspeed(pl_module, args):
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True
        },
        "fp16": {
            "enabled": True
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pl_module.torch_nn_module,
        model_parameters=pl_module.torch_nn_module.parameters(),
        config_params=deepspeed_config
    )
    return model_engine, optimizer

# Modify the main function to integrate DeepSpeed
def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_cfg = OmegaConf.to_object(CuboidEarthNet2021PLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0

    seed_everything(seed, workers=True)
    dataloader_dict = CuboidEarthNet2021PLModule.get_earthnet2021_dataloaders(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8, )
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidEarthNet2021PLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dataloader_dict["num_train_samples"],
        total_batch_size=total_batch_size,
    )

    pl_module = CuboidEarthNet2021PLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)

    # Initialize DeepSpeed
    model_engine, optimizer = initialize_deepspeed(pl_module, args)
    pl_module.torch_nn_module = model_engine.module  # Use the DeepSpeed model wrapper

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        pretrained_ckpt_name = pytorch_state_dict_name
        if not os.path.exists(os.path.join(pl_module.save_dir, "checkpoints", pretrained_ckpt_name)):
            warnings.warn(f"ckpt {pretrained_ckpt_name} not exists! Start training from epoch 0.")
            ckpt_path = None

        state_dict = torch.load(os.path.join(pl_module.save_dir, "checkpoints",pretrained_ckpt_name),
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)

    elif args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        pl_module.reset_test_epoch_count()
        for test_dataloader in dataloader_dict["test_dataloader"]:
            trainer.test(model=pl_module,
                         dataloaders=test_dataloader,
                         ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    train_dataloaders=dataloader_dict["train_dataloader"],
                    val_dataloaders=dataloader_dict["val_dataloader"],
                    ckpt_path=ckpt_path)
        state_dict = pl_ckpt_to_pytorch_state_dict(checkpoint_path=trainer.checkpoint_callback.best_model_path,
                                                   map_location=torch.device("cpu"),
                                                   delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        pl_module.reset_test_epoch_count()
        for test_dataloader in dataloader_dict["test_dataloader"]:
            trainer.test(ckpt_path="best",
                         dataloaders=test_dataloader, )

if __name__ == "__main__":
    main()

