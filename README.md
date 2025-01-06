TSformer: A Non-autoregressive Spatial-temporal Transformer for 30-day Ocean Eddy-Resolving Forecasting
TSformer is a non-autoregressive spatiotemporal model based on the Transformer architecture, focusing on processing multidimensional time series data (3DTS) and auxiliary information (Auxiliary Data). The model combines temporal and spatial feature extraction capabilities, making it suitable for various time series prediction tasks, such as ocean temperature and salinity forecasting.

Project Structure
TSformer/
├── data/                    # Dataset directory
├── model/                   # Model code
│   ├── tsformer.py          # TSformer model implementation
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loader
│   ├── metrics.py           # Evaluation metrics
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Dependency libraries
├── README.md                # Project description


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

Installation and Execution
pip install -r requirements.txt

Run inference:
python inference.py
