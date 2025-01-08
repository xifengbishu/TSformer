# TSformer: A Non-autoregressive Spatial-temporal Transformer for 30-day Ocean Eddy-Resolving Forecasting

Ocean forecasting is critical for various applications and is essential for understanding air-sea interactions, which contribute to mitigating the impacts of extreme events. State-of-the-art ocean numerical forecasting systems can offer lead times of up to 10 days with a spatial resolution of 10 kilometers, although they are computationally expensive. While data-driven forecasting models have demonstrated considerable potential and speed, they often primarily focus on spatial variations while neglecting temporal dynamics. This paper presents TSformer, a novel non-autoregressive spatiotemporal transformer designed for medium-range ocean eddy-resolving forecasting, enabling forecasts of up to 30 days in advance. We introduce an innovative hierarchical U-Net encoder-decoder architecture based on 3D Swin Transformer blocks, which extends the scope of local attention computation from spatial to spatiotemporal contexts to reduce accumulation errors. TSformer is trained on 28 years of homogeneous, high-dimensional 3D ocean reanalysis datasets, supplemented by three 2D remote sensing datasets for surface forcing. Based on the near-real-time operational forecast results from 2023, comparative performance assessments against in situ profiles and satellite observation data indicate that, TSformer exhibits forecast performance comparable to leading numerical ocean forecasting models while being orders of magnitude faster. Unlike autoregressive models, TSformer maintains 3D consistency in physical motion, ensuring long-term coherence and stability in extended forecasts. Furthermore, the TSformer model, which incorporates surface auxiliary observational data, effectively simulates the vertical cooling and mixing effects induced by Super Typhoon Saola.

---
## Introduction
The ocean is a complex system that is influenced by numerous factors, including surface forcing, ocean currents, and atmospheric conditions. The ocean plays a crucial role in influencing climate, weather, and human health. However, the current state-of-the-art numerical ocean forecasting systems are computationally expensive and cannot predict long-term trends. This paper presents TSformer, a novel non-autoregressive spatiotemporal transformer designed for medium-range ocean eddy-resolving forecasting. TSformer istrained on 28 years of homogeneous, high-dimensional 3D ocean re analysis datasets, supplemented by three 2D remote sensing datasets for surface forcing.

## Installation
To install the required packages, run the following command:
git clone https://github.com/xifengbishu/TSformer.git

cd TSformer
conda create -n TSformer python=3.7
conda activate TSformer
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c pytorch pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
conda install -c conda-forge opencv
conda install -c conda-forge seaborn
conda install -c conda-forge tensorboard
conda install -c conda-forge pyyaml

or

pip install -r requirements.txt

## Requirements
- Python 3.7 or higher
- PyTorch 1.7 or higher
- torch>=1.7
- numpy>=1.19
- pandas>=1.3
- scikit-learn>=0.24
- matplotlib>=3.3
- seaborn>=0.11
- tensorboard>=2.4
- torchvision>=0.8
- scikit-image>=0.18
- opencv-python>=4.5.62
---

## Input Data
The input data for TSformer consists of two types: 3DTS and auxiliary data.

Shared file via Baidu Netdisk: TSformer_Input
Link: https://pan.baidu.com/s/1bLsaKeMZAZJ6PE6lbN7v3w?pwd=4u7k

### 1. **3DTS (3D TS nowcast Data)**
   - **Shape**: `(batch_size, sequence_length, num_nodes, num_features)`
     - `batch_size`: Batch size.
     - `sequence_length`: Length of the time series (historical time steps).
     - `num_nodes`: Number of spatial nodes (e.g., sensors or geographical locations).
     - `num_features`: Number of features per time step (e.g., temperature, salinity, etc.).
   - **Example**: In ocean temperature data, `num_nodes` can represent different monitoring stations, and `num_features` can represent temperature, salinity, etc.

### 2. **Auxiliary Data**
   - **Type**: Static or dynamic auxiliary information.
     - **Static Data**: For example, geographical locations or category labels of nodes, with a shape of `(num_nodes, num_static_features)`.
     - **Dynamic Data**: For example, timestamp information (hour, day of the week, etc.), with a shape of `(batch_size, sequence_length, num_temporal_features)`.
   - **Purpose**: Auxiliary data enhances the model's contextual understanding and improves prediction performance.
  
   - **Example**: In ocean temperature data, `num_static_features` can represent geographical locations, and `num_temporal_features` can represent hour, day of the week, etc.
 - 
   - **Usage**: The auxiliary data can be used in the model's forward pass to provide contextual information, such as SLA, SST, WIND etc., which can improve the model's prediction performance.


 - 

---

## Model Architecture
### An overview of the proposed  TSformer model architecture
![alt text](<3D-Transformer-Unet network structure-1.jpg>)

## Model Components
### 3D Swin Transformer block
![alt text](<3D Swin Transformer block-1.jpg>)


## Run Inference

To run the inference script, use the following command:
python inference.py --model_path=path/to/your/model.pth 

```python
model = TSformer(
    num_nodes=100,                  # Number of spatial nodes
    in_features=29,                  # Number of input features
    out_features=26,                 # Number of output features
    seq_len=10,                     # Input sequence length
    pred_len=10,                    # Prediction sequence length
    d_model=64,                     # Hidden layer dimension of the Transformer
    nhead=4,                        # Number of attention heads in the Transformer
    num_encoder_layers=3,           # Number of encoder layers
    num_decoder_layers=3,           # Number of decoder layers
    dim_feedforward=256,            # Dimension of the feedforward layer
    dropout=0.1,                    # Dropout probability
    activation="relu",              # Activation function
    use_auxiliary=True,             # Whether to use auxiliary data
    auxiliary_dim=10,               # Dimension of auxiliary data features
)
```

## Training
To train the model, use the following command:
python train.py --model_path=path/to/your/model.pth

```python
model = TSformer(
    num_nodes=100,                  # Number of spatial nodes
    in_features=29,                  # Number of input features
    out_features=26,
    seq_len=10,
    pred_len=10,
    d_model=64,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    activation="relu",
    use_auxiliary=True,
    auxiliary_dim=10,
    device="cuda",
    learning_rate=0.001,
    batch_size=32,
    num_epochs=100,
    weight_decay=0.0001,
    early_stopping_patience=10,
    early_stopping_delta=0.001,
    early_stopping_metric="loss",
    early_stopping_mode="min",
    save_best_model=True,
    save_best_model_metric="loss",
    save_best_model_mode="min",
    save_best_model_path="path/to/your/best_model.pth",
    save_checkpoints=True,
    save_checkpoints_path="path/to/your/checkpoints/",
    save_checkpoints_interval=1,
    save_checkpoints_metric="loss",
    save_checkpoints_mode="min",

)
'''

