TSformer: A Non-autoregressive Spatial-temporal Transformer for 30-day Ocean Eddy-Resolving Forecasting
TSformer is a non-autoregressive spatiotemporal model based on the Transformer architecture, focusing on processing multidimensional time series data (3DTS) and auxiliary information (Auxiliary Data). The model combines temporal and spatial feature extraction capabilities, making it suitable for various time series prediction tasks, such as ocean temperature and salinity forecasting.

Input Data
1. 3DTS (3D Time Series Data)
Shape: (batch_size, sequence_length, num_nodes, num_features)

batch_size: Batch size.

sequence_length: Length of the time series (historical time steps).

num_nodes: Number of spatial nodes (e.g., sensors or geographical locations).

num_features: Number of features per time step (e.g., temperature, salinity, etc.).

Example: In ocean temperature data, num_nodes can represent different monitoring stations, and num_features can represent temperature, salinity, etc.

2. Auxiliary Data
Type: Static or dynamic auxiliary information.

Static Data: For example, geographical locations or category labels of nodes, with a shape of (num_nodes, num_static_features).

Dynamic Data: For example, timestamp information (hour, day of the week, etc.), with a shape of (batch_size, sequence_length, num_temporal_features).

Purpose: Auxiliary data enhances the model's contextual understanding and improves prediction performance.

Model Parameters
Below are the main parameter configurations for TSformer:
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

Installation and Execution
pip install -r requirements.txt
Run inference:
python inference.py
