---

# 🧠 Neural Network from Scratch — Handwritten Digits Classification

This repository contains a **simple neural network built from scratch (using only NumPy)** to classify handwritten digits from the **`scikit-learn` Digits Dataset**.
No high-level libraries like TensorFlow or PyTorch are used — this project is purely for **learning the fundamentals of neural networks** such as **forward propagation, backward propagation, and gradient descent**.

---

## 🚀 Features

* Implements a **2-layer feedforward neural network**
* **ReLU activation** in the hidden layer
* **Softmax activation** in the output layer
* One-hot encoding for labels
* Manual implementation of **gradient descent**
* Achieves **\~95% accuracy** on the test set

---

## 📂 Project Structure

```
neural_network_digits/
│
├── neural_network.py   # Main Python file (your code)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (optional)
```

---

## ⚙️ How It Works

### 1️⃣ **Dataset Preparation**

* Loads the digits dataset (`8x8` grayscale images, pixel values from 0–16)
* Normalizes features by dividing by `16`
* Splits into **train (80%)** and **test (20%)** sets

### 2️⃣ **Model Architecture**

| Layer        | Type  | Units          | Activation |
| ------------ | ----- | -------------- | ---------- |
| Input Layer  | Dense | 64 (8×8 image) | —          |
| Hidden Layer | Dense | 10             | ReLU       |
| Output Layer | Dense | 10             | Softmax    |

---

### 3️⃣ **Training Process**

* **Forward Propagation** → compute predictions
* **Backward Propagation** → calculate gradients manually
* **Gradient Descent** → update weights and biases iteratively

---

## 🖥️ Usage

### **1. Clone the Repository**

```bash
git clone https://github.com/vinayakkamatcodes/neural_netwrk.git
cd neural_netwrk
```

### **2. Install Dependencies**

```bash
pip install numpy scikit-learn
```

### **3. Run the Training Script**

```bash
python neural_network.py
```

---

## 📊 Sample Output

```
--- Starting Training ---
Iteration: 0
Accuracy: 11.25%
Iteration: 50
Accuracy: 72.80%
Iteration: 100
Accuracy: 85.90%
...
--- Training Complete ---

--- Testing on Test Set ---
Test Accuracy: 95.40%
```

---

## 🧠 Key Learnings

* Matrix multiplication for **forward propagation**
* Chain rule for **backpropagation**
* Gradient updates and **learning rate tuning**
* How neural networks "learn" from data

---

## 📌 Future Improvements

* Add more hidden layers for deeper architecture
* Implement **mini-batch gradient descent**
* Add **regularization** (L2 or dropout) to prevent overfitting
* Visualize **loss vs. iterations**

---


