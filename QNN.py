import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the device with 2 qubits.
n_qubits = 2
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def quantum_neural_net(inputs, weights):
    """
    Quantum neural network:
    - Embeds a 2D data point into the quantum state.
    - Applies 3 variational layers of parameterized rotations and entanglement.
    - Returns the expectation value of the Pauli-Z observable on qubit 0.
    """
    # 1. Encode the 2D input into the quantum state.
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. Apply 3 variational layers.
    for layer in range(n_layers):
        # Apply parameterized rotations on each qubit.
        for i in range(n_qubits):
            qml.Rot(weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i)
        # Entangle the qubits with a CNOT gate.
        qml.CNOT(wires=[0, 1])
    
    # 3. Measure the expectation value of the PauliZ operator on qubit 0.
    return qml.expval(qml.PauliZ(0))

# Initialize weights randomly with gradient tracking enabled.
np.random.seed(42)
weights = np.random.randn(n_layers, n_qubits, 3, requires_grad=True)

# Define a toy 2D dataset and target outputs.
data = np.array([[0.1,  0.2],
                 [0.4,  0.2],
                 [0.3, -0.1],
                 [-0.2, -0.3]])
# Target outputs are chosen to be in the range [-1, 1]
targets = np.array([1, 1, -1, -1])

def cost(weights, data, targets):
    """Mean squared error loss over the dataset."""
    predictions = [quantum_neural_net(x, weights) for x in data]
    return np.mean((np.array(predictions) - targets) ** 2)

# Choose a gradient descent optimizer.
opt = qml.GradientDescentOptimizer(stepsize=0.1)
epochs = 200

# Training loop: record the loss for each epoch.
loss_history = []
for epoch in range(epochs):
    weights, loss_val = opt.step_and_cost(lambda w: cost(w, data, targets), weights)
    loss_history.append(loss_val)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss_val}")

# Plot the training loss vs. epoch.
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs. Epoch for the Standard QNN")
plt.grid(True)
plt.show()
