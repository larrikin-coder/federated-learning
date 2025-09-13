import requests
import torch
import pandas as pd
from models import LinearRegressionModel
from utils import local_train
import argparse
import torch.onnx

SERVER_URL = "http://127.0.0.1:5000"

# -------------------------
# CLI Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
args = parser.parse_args()
DATASET_PATH = args.dataset

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(DATASET_PATH)
x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
INPUT_DIM = x.shape[1]

# -------------------------
# Get global model from server
# -------------------------
resp = requests.get(f"{SERVER_URL}/get_model")
global_state = {k: torch.tensor(v) for k, v in resp.json().items()}

model = LinearRegressionModel(input_dim=INPUT_DIM)
model.load_state_dict(global_state)

# -------------------------
# Train locally
# -------------------------
new_state = local_train(model, x, y, epoch=5)

# -------------------------
# Save updated model (PyTorch + ONNX)
# -------------------------
torch.save(model.state_dict(), "global_model.pt")

dummy_input = torch.randn(1, INPUT_DIM)
torch.onnx.export(
    model,
    dummy_input,
    "global_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("✅ Saved model in both PyTorch (.pt) and ONNX (.onnx) formats")

# -------------------------
# Send updated weights to server
# -------------------------
resp = requests.post(
    f"{SERVER_URL}/send_update",
    json={k: v.tolist() for k, v in new_state.items()}
)
print(resp.json())
