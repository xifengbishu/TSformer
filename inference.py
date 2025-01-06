import torch
from model import TSformer

# Load the pre-trained model
model = TSformer(...)
model.load_state_dict(torch.load("tsformer_model.pth"))
model.eval()

# Prepare input data
batch_size = 32
seq_len = 10
num_nodes = 100
num_features = 29
auxiliary_dim = 10

# Generate example data
input_3dts = torch.randn(batch_size, seq_len, num_nodes, num_features)  # 3DTS data
auxiliary_data = torch.randn(batch_size, seq_len, auxiliary_dim)        # Auxiliary data

# Inference
with torch.no_grad():
    predictions = model(input_3dts, auxiliary_data)  # Output shape: (batch_size, pred_len, num_nodes, num_features)

print("Predictions:", predictions.shape)