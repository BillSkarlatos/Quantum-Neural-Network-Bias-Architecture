import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# --- Setup Parameters ---
n_main_qubits = 2            # Main qubits for 2D data
n_layers = 3                 # Number of variational layers
total_wires = n_main_qubits + 1  # Extra wire for the bias (topological) qubit
bias_wire = n_main_qubits    # Designate the extra wire for the bias qubit

# --- Define Quantum Device ---
dev = qml.device("default.qubit", wires=total_wires)

# --- Define the Topological QNN Circuit ---
@qml.qnode(dev, interface="autograd")
def quantum_neural_net_with_bias(inputs, weights, bias_weights):
    """
    Topological QNN with an external bias (topological) qubit.
    
    Args:
      inputs (array): A 2D data point.
      weights (array): Variational weights for the main rotations (shape: [n_layers, n_main_qubits, 3]).
      bias_weights (array): Bias parameters for controlled gates (shape: [n_layers, n_main_qubits]).
      
    Returns:
      Expectation value of PauliZ on main qubit 0.
    """
    # 1. Embed the 2D input into the main qubits.
    qml.templates.AngleEmbedding(inputs, wires=range(n_main_qubits))
    
    # 2. Initialize the bias qubit (simulate a topologically nontrivial state).
    qml.Hadamard(wires=bias_wire)
    
    # 3. Apply the variational layers with bias.
    for layer in range(n_layers):
        for i in range(n_main_qubits):
            # Apply parameterized rotation on the main qubit.
            qml.Rot(weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i)
            # Apply a controlled RY rotation: bias qubit controls the rotation.
            controlled_RY = qml.ctrl(qml.RY, control=bias_wire)
            controlled_RY(bias_weights[layer, i], wires=i)
        # Entangle the main qubits.
        qml.CNOT(wires=[0, 1])
    
    # 4. Return the expectation value of PauliZ on main qubit 0.
    return qml.expval(qml.PauliZ(0))

# --- Topology Update Function ---
def update_topology(epoch, n_layers, n_main_qubits):
    """
    Update bias parameters to simulate a topology change per epoch.
    Here, a sinusoidal function is used to generate new bias values.
    
    Args:
      epoch (int): The current epoch.
      n_layers (int): Number of layers.
      n_main_qubits (int): Number of main qubits.
    
    Returns:
      Array with new bias parameters (shape: [n_layers, n_main_qubits]).
    """
    total_elements = n_layers * n_main_qubits
    new_params = np.sin(np.linspace(epoch, epoch + total_elements, total_elements))
    return new_params.reshape(n_layers, n_main_qubits)

# --- Define the Dataset ---
data = np.array([[0.1,  0.2],
                 [0.4,  0.2],
                 [0.3, -0.1],
                 [-0.2, -0.3]])
# Targets in the range [-1, 1]
targets = np.array([1, 1, -1, -1])

# --- Initialize Parameters ---
np.random.seed(42)
weights = np.random.randn(n_layers, n_main_qubits, 3, requires_grad=True)
bias_weights = np.random.randn(n_layers, n_main_qubits)

# --- Define the Cost Function ---
def cost(weights, bias_weights, data, targets):
    predictions = [quantum_neural_net_with_bias(x, weights, bias_weights) for x in data]
    return np.mean((np.array(predictions) - targets)**2)

# --- Setup Adam Optimizer and Training ---
opt = qml.AdamOptimizer(stepsize=0.05)
epochs = 200
loss_history = []

for epoch in range(epochs):
    # Update the topology (bias parameters) each epoch.
    bias_weights = update_topology(epoch, n_layers, n_main_qubits)
    weights, loss_val = opt.step_and_cost(lambda w: cost(w, bias_weights, data, targets), weights)
    loss_history.append(loss_val)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss_val}")

# --- Graph the Training Loss ---
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimized Training Loss vs. Epoch for the Topological QNN (Adam Optimizer)")
plt.grid(True)
plt.show()
