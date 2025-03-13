import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's autograd-enabled NumPy
import numpy as np               # Standard NumPy for dataset generation and plotting
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# =============================================================================
# 1. Generate the Dataset (Concentric Circles)
# =============================================================================
def generate_annulus_points(N, inner_radius, outer_radius):
    """
    Generates N points uniformly distributed within an annulus defined
    by inner_radius and outer_radius.
    """
    # Sample radius uniformly over the area
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, N))
    theta = np.random.uniform(0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y]).T

N = 100  # samples per class

# Class 0: points inside a circle of radius 1.
data0 = generate_annulus_points(N, 0.0, 1.0)
# Class 1: points in an annulus between radius 1 and 2.
data1 = generate_annulus_points(N, 1.0, 2.0)
# Class 2: points in an annulus between radius 2 and 3.
data2 = generate_annulus_points(N, 2.0, 3.0)

# Combine data and labels
X = np.vstack([data0, data1, data2])
y = np.hstack([np.zeros(N), np.ones(N), 2 * np.ones(N)])

# =============================================================================
# 2. Define the Hybrid Quantum Neural Network (QNN)
# =============================================================================
# Quantum circuit settings
n_qubits = 2
n_layers = 1  # number of variational layers

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def quantum_circuit(features, weights):
    """
    Encodes 2 features into 2 qubits using RX rotations,
    applies a variational layer (BasicEntanglerLayers),
    and returns expectation values of PauliZ.
    """
    # Encode features on each qubit
    for i in range(n_qubits):
        qml.RX(features[i], wires=i)
    # Apply a variational layer
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qnn_model(features, q_weights, c_weights, c_bias):
    """
    Hybrid model: passes the features through the quantum circuit then
    through a linear classical layer to obtain 3 class logits.
    """
    qc_out = quantum_circuit(features, q_weights)  # shape: (2,)
    logits = pnp.dot(c_weights, qc_out) + c_bias    # shape: (3,)
    return logits

def qnn_softmax(logits):
    exps = pnp.exp(logits - pnp.max(logits))
    return exps / pnp.sum(exps)

def qnn_cross_entropy_loss(logits, label):
    """
    Compute cross-entropy loss for one sample.
    """
    probs = qnn_softmax(logits)
    return -pnp.log(probs[int(label)] + 1e-10)

def qnn_cost(params, X_data, y_data):
    """
    Compute the average cross-entropy loss over the dataset.
    Uses a list comprehension to ensure gradients flow properly.
    """
    loss_list = [qnn_cross_entropy_loss(
                    qnn_model(features, params["q_weights"], params["c_weights"], params["c_bias"]),
                    label)
                  for features, label in zip(X_data, y_data)]
    return pnp.mean(pnp.array(loss_list))

def qnn_accuracy(params, X_data, y_data):
    """
    Compute classification accuracy over the dataset.
    """
    correct = sum([1 if pnp.argmax(qnn_model(features, params["q_weights"], params["c_weights"], params["c_bias"])) == int(label) else 0
                   for features, label in zip(X_data, y_data)])
    return correct / len(X_data)

# Convert dataset to PennyLane's NumPy arrays for QNN training
X_qnn = pnp.array(X)
y_qnn = pnp.array(y)

# Initialize QNN parameters
np.random.seed(42)  # for reproducibility
q_weights = pnp.array(np.random.randn(n_layers, n_qubits), requires_grad=True)
c_weights = pnp.array(np.random.randn(3, n_qubits), requires_grad=True)
c_bias    = pnp.array(np.random.randn(3), requires_grad=True)
params_qnn = {"q_weights": q_weights, "c_weights": c_weights, "c_bias": c_bias}

# Set up the optimizer (Adam) for the QNN
opt = qml.AdamOptimizer(stepsize=0.1)
epochs = 200  # number of epochs (same for both QNN and MLP)
loss_history_qnn = []
acc_history_qnn = []

print("Starting QNN training...")
for epoch in range(epochs):
    params_qnn = opt.step(lambda p: qnn_cost(p, X_qnn, y_qnn), params_qnn)
    current_loss = qnn_cost(params_qnn, X_qnn, y_qnn)
    current_acc = qnn_accuracy(params_qnn, X_qnn, y_qnn)
    loss_history_qnn.append(current_loss)
    acc_history_qnn.append(current_acc)
    if (epoch + 1) % 20 == 0:
        print(f"QNN Epoch {epoch+1:3d}: Loss = {current_loss:.4f}, Accuracy = {current_acc*100:.2f}%")

print("\nQuantum Neural Network Training complete.")
print("Total QNN epochs: ", epochs)
print("Quantum circuit: {} qubits with {} variational layer(s).".format(n_qubits, n_layers))
print("Classical output layer: {} neurons.".format(params_qnn["c_weights"].shape[0]))

# =============================================================================
# 3. Train the Classical MLP Neural Network for the Same Number of Epochs
# =============================================================================
# We use partial_fit to update the MLP one epoch at a time.
mlp = MLPClassifier(hidden_layer_sizes=(20,),
                    activation='relu',
                    warm_start=True,  # allows incremental learning
                    max_iter=1,
                    random_state=42)
classes = np.unique(y)
# The first call must include the classes parameter.
mlp.partial_fit(X, y, classes=classes)
mlp_loss_history = [mlp.loss_]
mlp_acc_history = [mlp.score(X, y)]
for epoch in range(1, epochs):
    mlp.partial_fit(X, y)
    mlp_loss_history.append(mlp.loss_)
    mlp_acc_history.append(mlp.score(X, y))
    if (epoch + 1) % 20 == 0:
        print(f"MLP Epoch {epoch+1:3d}: Loss = {mlp.loss_:.4f}, Accuracy = {mlp.score(X, y)*100:.2f}%")

print("\nMLP Training complete.")
print("Total MLP epochs: ", epochs)
print("MLP Hidden layer sizes: ", mlp.hidden_layer_sizes)

# =============================================================================
# 4. Compute Decision Boundaries for QNN and MLP
# =============================================================================
# Define a mesh grid over the feature space.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))

# Decision boundary for QNN
Z_qnn = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        sample = pnp.array([xx[i, j], yy[i, j]])
        logits = qnn_model(sample, params_qnn["q_weights"], params_qnn["c_weights"], params_qnn["c_bias"])
        Z_qnn[i, j] = int(pnp.argmax(logits))

# Decision boundary for MLP
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_mlp = mlp.predict(grid_points)
Z_mlp = Z_mlp.reshape(xx.shape)

# =============================================================================
# 5. Plot and Compare the Decision Boundaries
# =============================================================================
markers = {0: '+', 1: 'x', 2: 'o'}
colors = ['red', 'green', 'blue']

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# QNN Decision Boundary Plot
axs[0].contourf(xx, yy, Z_qnn, alpha=0.3, cmap=plt.cm.coolwarm)
for class_val in np.unique(y):
    idx = np.where(y == class_val)
    axs[0].scatter(X[idx, 0], X[idx, 1],
                   marker=markers[int(class_val)],
                   color=colors[int(class_val)],
                   edgecolor='k',
                   s=100,
                   label=f"Class {int(class_val)}")
axs[0].set_title("Quantum Neural Network Decision Boundary")
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")
axs[0].legend()

# MLP Decision Boundary Plot
axs[1].contourf(xx, yy, Z_mlp, alpha=0.3, cmap=plt.cm.coolwarm)
for class_val in np.unique(y):
    idx = np.where(y == class_val)
    axs[1].scatter(X[idx, 0], X[idx, 1],
                   marker=markers[int(class_val)],
                   color=colors[int(class_val)],
                   edgecolor='k',
                   s=100,
                   label=f"Class {int(class_val)}")
axs[1].set_title("MLP Decision Boundary")
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")
axs[1].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 6. Plot Training Accuracy Comparison
# =============================================================================
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), np.array(acc_history_qnn)*100, label="QNN Accuracy")
plt.plot(range(epochs), np.array(mlp_acc_history)*100, label="MLP Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy vs Epochs")
plt.legend()
plt.show()
