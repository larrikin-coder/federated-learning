from flask import Flask, request, jsonify
import torch
from models import LinearRegressionModel
from utils import average_models

app = Flask(__name__)

# Global model
input_dim = 2
global_model = LinearRegressionModel(input_dim=input_dim)
received_weights = []

@app.route("/get_model", methods=["GET"])
def get_model():
    """Send the current global model state_dict to a client."""
    return jsonify({k: v.tolist() for k, v in global_model.state_dict().items()})

@app.route("/send_update", methods=["POST"])
def receive_update():
    """Receive client model weights."""
    global received_weights, global_model
    data = request.get_json()
    state_dict = {k: torch.tensor(v) for k, v in data.items()}
    received_weights.append(state_dict)

    # Aggregate if multiple clients have sent updates
    if len(received_weights) >= 2:  # <-- adjust to number of clients
        new_state = average_models(received_weights)
        global_model.load_state_dict(new_state)
        received_weights = []  # reset
        return jsonify({"status": "global model updated"})
    else:
        return jsonify({"status": "update received, waiting for more clients"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
