import torch
from model import TSformer
import warnings
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from torchsummary import summary
from pathlib import Path
import netCDF4 as nc
from datetime import datetime, timedelta


# Load the pre-trained model
model = torch.load('Tmodule.pth', map_location=torch.device('cpu'))  # 加载Smodule.pth模型
model.torch_nn_module.cpu()
model.torch_nn_module.eval()

# Prepare input data
batch_size = 1
seq_len = 10
num_nodes = 100
num_features = 29
auxiliary_dim = 10

# Generate example data
input_3dts = read_3dts(batch_size, seq_len, num_nodes, num_features)  # 3DTS data
auxiliary_data = read_aux(batch_size, seq_len, auxiliary_dim)        # Auxiliary data

print("Input 3DTS data:", input_3dts.shape)
print("Auxiliary data:", auxiliary_data.shape)
print("Model:", model)
print("Model parameters:", model.parameters())
print("Model device:", next(model.parameters()).device)

# Inference
with torch.no_grad():
    predictions = model(input_3dts, auxiliary_data)  # Output shape: (batch_size, pred_len, num_nodes, num_features)

print("Predictions:", predictions.shape)