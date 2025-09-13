import pandas as pd
import numpy as np
import onnxruntime as ort   # <-- make sure this is imported

# -------------------------
# Load ONNX model
# -------------------------
session = ort.InferenceSession("global_model.onnx")

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("dataset1.csv")
x = df.iloc[:, :-1].values.astype(np.float32)  # features only

# -------------------------
# Run inference
# -------------------------
predictions = session.run(None, {"input": x})[0]

print("Predictions shape:", predictions.shape)
print("First 5 predictions:", predictions[:5].squeeze())
y_true = df.iloc[:, -1].values.astype(np.float32)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_true, predictions)
r2 = r2_score(y_true, predictions)

print("MSE:", mse)
print("R²:", r2)
