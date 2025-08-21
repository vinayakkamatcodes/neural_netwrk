import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- Data Loading and Preprocessing ---
# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transpose the data matrices to have shape (number of features, number of examples)
# (64, num_examples)
X_train = X_train.T
X_test = X_test.T

# Normalize pixel values to be between 0 and 1
# The load_digits dataset has pixel values from 0 to 16.
X_train = X_train / 16.0
X_test = X_test / 16.0

# Reshape y to be (1, num_examples)
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

# Get dimensions of the data
n_x = X_train.shape[0]  # Number of features (64, since 8x8 image flattened)
m_train = X_train.shape[1] # Number of training examples
m_test = X_test.shape[1]   # Number of test examples

# Define our network architecture
n_h = 10  # Number of neurons in the hidden layer (you can experiment with this)
n_y = 10  # Number of neurons in the output layer (10 digits: 0-9)

print(f"Number of features (n_x): {n_x}")
print(f"Number of training examples (m_train): {m_train}")
print(f"Number of test examples (m_test): {m_test}")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")


# --- Parameter Initialization ---
def init_params():
    """
    Initializes the weights and biases for the neural network.
    Weights are initialized with small random numbers from a standard normal distribution,
    and biases are also initialized with small random numbers.
    
    Returns:
        tuple: W1, b1, W2, b2 - The initialized weight matrices and bias vectors.
    """
    # W1: (n_h, n_x), b1: (n_h, 1)
    W1 = np.random.randn(n_h, n_x) * 0.01  # Using * 0.01 for small initial weights
    b1 = np.zeros((n_h, 1))

    # W2: (n_y, n_h), b2: (n_y, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    


    
    return W1, b1, W2, b2

# --- Activation Functions ---
def ReLU(Z):
    """
    Implements the Rectified Linear Unit (ReLU) activation function.
    
    Arguments:
    Z -- The output of the linear layer, a numpy array of any shape.
    
    Returns:
    A -- The output of ReLU(Z), same shape as Z.
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    Implements the Softmax activation function.
    It converts a vector of numbers into a probability distribution.
    
    Arguments:
    Z -- The output of the linear layer, a numpy array of shape (n_y, m).
    
    Returns:
    A -- The output of softmax(Z), a probability distribution over classes.
         Each column sums to 1.
    """
    # Subtracting the max for numerical stability to prevent overflow
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A

# --- Forward Propagation ---
def forward_prop(W1, b1, W2, b2, X):
    """
    Implements the forward propagation for our two-layer neural network.
    
    Arguments:
    W1, b1, W2, b2 -- The parameters (weights and biases) of the model.
    X -- The input data of shape (n_x, m).
    
    Returns:
    Z1, A1, Z2, A2 -- The intermediate and final values computed during forward propagation.
                       These are needed for the backward pass.
    """
    # Hidden Layer calculations
    Z1 = W1.dot(X) + b1  # Linear combination
    A1 = ReLU(Z1)        # Activation function (ReLU)
    
    # Output Layer calculations
    Z2 = W2.dot(A1) + b2 # Linear combination
    A2 = softmax(Z2)     # Activation function (Softmax)
    
    return Z1, A1, Z2, A2

# --- One-Hot Encoding ---
def one_hot(Y):
    """
    Converts a vector of integer labels into a one-hot encoded matrix.
    
    Arguments:
    Y -- The label vector of shape (1, m), containing integer class labels.
    
    Returns:
    one_hot_Y -- A one-hot encoded matrix of shape (num_classes, m).
    """
    # Create an (num_examples) x (num_classes) matrix of zeros
    # Y.max() + 1 gives the total number of unique classes (0-9, so 10 classes)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Corrected syntax: (Y.size, Y.max() + 1)
    
    # Set the element at the correct column (label) to 1 for each example
    # np.arange(Y.size) generates indices for each example
    one_hot_Y[np.arange(Y.size), Y] = 1
    
    # Transpose to get the desired shape (num_classes, num_examples)
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# --- Backward Propagation ---
def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Implements the backward propagation (backprop) for our two-layer network.
    Calculates the gradients of the loss with respect to each parameter.
    
    Arguments:
    Z1, A1, Z2, A2 -- Values from the forward propagation.
    W2 -- Weight matrix from the output layer (needed for dZ1 calculation).
    X -- The input data.
    Y -- The true labels (used to calculate loss derivative).
    
    Returns:
    dW1, db1, dW2, db2 -- The gradients for each weight matrix and bias vector.
    """
    m = X.shape[1] # Number of examples

    # Convert true labels to one-hot encoding for comparison with A2
    one_hot_Y = one_hot(Y)
    
    # Gradients for the Output Layer (Softmax + Cross-Entropy)
    dZ2 = A2 - one_hot_Y # Derivative of loss wrt Z2
    dW2 = 1/m * dZ2.dot(A1.T) # Gradient for W2
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True) # Gradient for b2 (sum across examples)
    
    # Gradients for the Hidden Layer (ReLU)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0) # Chain rule: (W2_transpose * dZ2) * ReLU_derivative(Z1)
                                   # (Z1 > 0) acts as 1 where Z1 > 0, and 0 otherwise
    dW1 = 1/m * dZ1.dot(X.T) # Gradient for W1
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True) # Gradient for b1
    
    return dW1, db1, dW2, db2

# --- Parameter Update ---
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Updates the neural network parameters using gradient descent.
    
    Arguments:
    W1, b1, W2, b2 -- The current parameters.
    dW1, db1, dW2, db2 -- The gradients of the parameters.
    alpha -- The learning rate, controls the step size of updates.
    
    Returns:
    W1, b1, W2, b2 -- The updated parameters after one step of gradient descent.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    return W1, b1, W2, b2

# --- Prediction and Accuracy ---
def get_predictions(A2):
    """
    Gets the class predictions from the output probabilities.
    
    Arguments:
    A2 -- The output probabilities from the softmax layer, shape (n_y, m).
    
    Returns:
    predictions -- A 1D array of predicted class labels (0-9).
    """
    return np.argmax(A2, 0) # Returns the index (class label) of the max probability for each example

def get_accuracy(predictions, Y):
    """
    Calculates the accuracy of the predictions against the true labels.
    
    Arguments:
    predictions -- A 1D array of predicted class labels.
    Y -- The true labels, shape (1, m).
    
    Returns:
    accuracy -- The percentage of correct predictions (float between 0 and 1).
    """
    # print(f"Predictions shape: {predictions.shape}, Y shape: {Y.shape}") # Debugging aid
    # print(f"First 10 predictions: {predictions[:10]}")
    # print(f"First 10 true labels: {Y[0, :10]}")
    return np.sum(predictions == Y) / Y.size

# --- Gradient Descent (Training Loop) ---
def gradient_descent(X, Y, alpha, iterations):
    """
    Performs gradient descent to train the neural network.
    This function orchestrates the entire training process:
    initialization -> forward prop -> backward prop -> parameter update.
    
    Arguments:
    X -- The input training data (n_x, m_train).
    Y -- The true labels for the training data (1, m_train).
    alpha -- The learning rate.
    iterations -- The number of training iterations (epochs).
    
    Returns:
    W1, b1, W2, b2 -- The final trained parameters.
    """
    # 1. Initialize parameters at the start of training
    W1, b1, W2, b2 = init_params()
    
    print("\n--- Starting Training ---")
    for i in range(1, iterations + 1): # Start iteration count from 1 for better display
        # 2. Forward Propagation: Calculate predictions and intermediate values
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # 3. Backward Propagation: Calculate gradients based on prediction error
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        
        # 4. Update Parameters: Adjust weights and biases
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # 5. Print progress (optional)
        if i % 100 == 0 or i == 1: # Print every 100 iterations and at the very first iteration
            predictions = get_predictions(A2)
            current_accuracy = get_accuracy(predictions, Y) * 100
            print(f"Iteration: {i}, Training Accuracy: {current_accuracy:.2f}%")
            
    print("--- Training Complete ---")
    return W1, b1, W2, b2

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure X_train and y_train are correctly prepared (already done above)

    # Train the model
    # You can adjust alpha (learning rate) and iterations (epochs) to see different results.
    # A learning rate of 0.1 and 1000-2000 iterations is a good starting point.
    learning_rate = 0.10
    num_iterations = 2000 # Increased iterations for potentially better accuracy

    final_W1, final_b1, final_W2, final_b2 = gradient_descent(X_train, y_train, learning_rate, num_iterations)

    # Test the model on the unseen test set
    print("\n--- Testing on Test Set ---")
    # Perform forward propagation on the test data using the trained parameters
    _, _, _, A2_test = forward_prop(final_W1, final_b1, final_W2, final_b2, X_test)
    
    # Get predictions and calculate accuracy on the test set
    test_predictions = get_predictions(A2_test)
    test_accuracy = get_accuracy(test_predictions, y_test) * 100
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Optional: Display a sample prediction
    # import matplotlib.pyplot as plt
    # index = 0 # Change this to view different examples
    # plt.imshow(X_test[:, index].reshape(8, 8), cmap='gray')
    # plt.title(f"Predicted: {test_predictions[index]}, Actual: {y_test[0, index]}")
    # plt.show()
