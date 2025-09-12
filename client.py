# client.py
import requests
import torch
import pandas as pd
from models import LinearRegressionModel
from utils import local_train

# ------------------------
# CONFIG
# ------------------------
SERVER_URL = "http://127.0.0.1:5000"
# INPUT_DIM =    # set according to dataset features
DATASET_PATH = "dataset1.csv"  # change to dataset2.csv for client 2

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(DATASET_PATH)

# Assume last column is target, rest are features
x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
INPUT_DIM = x.shape[1]
# ------------------------
# Get global model from server
# ------------------------
resp = requests.get(f"{SERVER_URL}/get_model")
global_state = {k: torch.tensor(v) for k, v in resp.json().items()}

model = LinearRegressionModel(input_dim=INPUT_DIM)
model.load_state_dict(global_state)

# ------------------------
# Train locally
# ------------------------
new_state = local_train(model, x, y, epoch=5)

# ------------------------
# Send updated weights to server
# ------------------------
resp = requests.post(f"{SERVER_URL}/send_update",
                     json={k: v.tolist() for k, v in new_state.items()})

print(resp.json())
