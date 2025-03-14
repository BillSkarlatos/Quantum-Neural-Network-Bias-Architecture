import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# ----------------------------
# Visualization Functions
# ----------------------------

def visualize_1qubit_image(p, C, img_height=100, base_color=np.array([0.8, 0.2, 0.2]), steepness=15):
    """
    Create a 2D RGB image for a 1-qubit state.
    
    Parameters:
      p (float): Probability of measuring |0> (0 <= p <= 1). The left side represents |0> and the right side |1>.
      C (int): Clarity index. The number of columns is 10^C.
      img_height (int): Height of the image in pixels.
      base_color (np.array): Base RGB color (values in [0,1]) to use.
      steepness (float): Controls the sharpness of the brightness transition.
      
    Returns:
      img (np.array): An array of shape (img_height, 10^C, 3) representing the RGB image.
    """
    N = 10 ** C  # total number of columns
    # Create normalized column indices (0 to 1)
    x = np.linspace(0, 1, N)
    # Use a logistic (sigmoid) function to get a sharper transition near the middle:
    sigmoid = 1 / (1 + np.exp(-steepness * (x - 0.5)))
    # For a 1-qubit state, we want the left brightness to be p (|0>) and right brightness 1-p (|1>).
    brightness = p + (1 - 2 * p) * sigmoid
    # Create a 2D image by repeating the 1D brightness profile vertically.
    brightness_image = np.tile(brightness, (img_height, 1))
    # Multiply the brightness image by the base_color (applied per channel).
    img = brightness_image[:, :, np.newaxis] * base_color[np.newaxis, np.newaxis, :]
    return img

def visualize_2qubit_image(probs, C, img_size=None, steepness=15, base_color=np.array([0.8, 0.2, 0.2])):
    """
    Create a 2D RGB image representing a 2-qubit state's measurement probabilities.
    
    The horizontal axis corresponds to qubit 0 (left ~ |0>, right ~ |1>), 
    and the vertical axis corresponds to qubit 1 (top ~ |0>, bottom ~ |1>).
    
    At normalized coordinates (u,v), the brightness is defined as:
    
        B(u,v) = p00 * f(u)*f(v)
               + p01 * (1 - f(u))*f(v)
               + p10 * f(u)*(1 - f(v))
               + p11 * (1 - f(u))*(1 - f(v))
    
    where f(t) = 1/(1 + exp(steepness*(t - 0.5))).
    
    Parameters:
      probs (list/array): Four probabilities [p00, p01, p10, p11].
      C (int): Clarity index. If img_size is not provided, the resolution is 10^C x 10^C.
      img_size (int): Image resolution (width and height in pixels). If None, set to 10^C.
      steepness (float): Controls the sharpness of the brightness transition.
      base_color (np.array): Base RGB color for the visualization.
      
    Returns:
      img (np.array): A 3D array (shape: [img_size, img_size, 3]) representing the RGB image.
    """
    N = 10 ** C if img_size is None else img_size
    # Create normalized coordinate arrays for horizontal (u) and vertical (v) axes.
    u = np.linspace(0, 1, N)
    v = np.linspace(0, 1, N)
    U, V = np.meshgrid(u, v)
    
    # Logistic functions that transition sharply near 0.5.
    fU = 1 / (1 + np.exp(steepness * (U - 0.5)))
    fV = 1 / (1 + np.exp(steepness * (V - 0.5)))
    
    p00, p01, p10, p11 = probs
    # Compute brightness at each (u,v) coordinate.
    brightness = (p00 * fU * fV +
                  p01 * (1 - fU) * fV +
                  p10 * fU * (1 - fV) +
                  p11 * (1 - fU) * (1 - fV))
    
    # Form the RGB image by multiplying the brightness map by the base_color.
    img = brightness[:, :, np.newaxis] * base_color[np.newaxis, np.newaxis, :]
    return img

# ----------------------------
# Adaptive Main Function
# ----------------------------

def main(num_qubits=1, clarity_index=3, img_height=100, steepness=15, base_color=np.array([0.8, 0.2, 0.2])):
    """
    Adaptive visualization for 1-qubit and 2-qubit systems.
    
    Depending on num_qubits, this function builds and simulates a Qiskit circuit,
    extracts the measurement probabilities, and visualizes the result using the appropriate method.
    
    Parameters:
      num_qubits (int): 1 or 2.
      clarity_index (int): Clarity index C (resolution = 10^C).
      img_height (int): For 1-qubit visualization, height of the image in pixels.
      steepness (float): Controls the sharpness of the brightness transition.
      base_color (np.array): Base RGB color.
    """
    backend = Aer.get_backend('statevector_simulator')
    
    if num_qubits == 1:
        # Build a 1-qubit circuit.
        qc = QuantumCircuit(1)
        # Apply an RY rotation to prepare a superposition (adjust theta as desired).
        theta = np.pi / 3  # For example, theta = pi/3 yields p ≈ cos^2(theta/2) ~ 0.75.
        qc.ry(theta, 0)
        
        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled).result()
        statevector = result.get_statevector(qc)
        
        # Compute probability for |0> and |1>.
        p0 = np.abs(statevector[0])**2
        p1 = np.abs(statevector[1])**2
        print("1-Qubit Measurement Probabilities:")
        print("  p(|0>) =", p0)
        print("  p(|1>) =", p1)
        
        # Generate the visualization image.
        img = visualize_1qubit_image(p0, clarity_index, img_height=img_height, base_color=base_color, steepness=steepness)
        
        plt.figure(figsize=(10, 2))
        plt.imshow(img, aspect='auto')
        plt.title("1-Qubit Topological Qubit Visualization")
        plt.axis('off')
        plt.show()
        
    elif num_qubits == 2:
        # Build a more complex 2-qubit circuit.
        qc = QuantumCircuit(2)
        # First layer: prepare a superposition on qubit 0 and entangle with qubit 1.
        qc.h(0)
        qc.cx(0, 1)
        # Additional rotation on qubit 1 to vary amplitudes.
        qc.ry(np.pi/4, 1)
        
        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled).result()
        statevector = result.get_statevector(qc)
        
        # Compute the measurement probabilities.
        p00 = np.abs(statevector[0])**2  # |00>
        p01 = np.abs(statevector[1])**2  # |01>
        p10 = np.abs(statevector[2])**2  # |10>
        p11 = np.abs(statevector[3])**2  # |11>
        print("2-Qubit Measurement Probabilities:")
        print("  p(|00>) =", p00)
        print("  p(|01>) =", p01)
        print("  p(|10>) =", p10)
        print("  p(|11>) =", p11)
        
        probs = [p00, p01, p10, p11]
        # For 2-qubit visualization, use a square image.
        img_size = 10 ** clarity_index  # resolution: 10^C x 10^C
        img = visualize_2qubit_image(probs, clarity_index, img_size=img_size, steepness=steepness, base_color=base_color)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img, aspect='auto')
        plt.title("2-Qubit Topological Qubit Visualization")
        plt.axis('off')
        plt.show()
    else:
        print("This function supports only 1 or 2 qubit systems.")

# ----------------------------
# Run the Adaptive Visualization
# ----------------------------

# Set num_qubits to 1 or 2
num_qubits = 2   # Change to 1 for 1-qubit visualization, 2 for 2-qubit
main(num_qubits=num_qubits, clarity_index=3, img_height=100, steepness=15, base_color=np.array([0.8, 0.2, 0.2]))
