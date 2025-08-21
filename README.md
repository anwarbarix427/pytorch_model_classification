**PyTorch Model Classification 🚀**
  
This repository contains PyTorch implementations of a linear regression model, a binary classification model, and two multi-class classification models. It is designed for learning and experimenting with supervised learning tasks using PyTorch. The code provides clear workflows for building, training, testing, and evaluating these models on synthetic datasets generated with scikit-learn.
Overview
**The project serves as an educational resource for understanding neural network implementations in PyTorch. It includes:**

A linear regression model to predict continuous values.
A binary classification model for classifying points into two classes using the make_circles dataset.
Two multi-class classification models for classifying points into multiple classes using make_blobs and make_moons datasets.
Helper functions for visualization (e.g., decision boundaries) and evaluation metrics (e.g., accuracy).

Each script is self-contained, with synthetic data generation, model training, and evaluation steps.
Features

Linear Regression: Predicts continuous outputs using a simple nn.Module with a single layer.
Binary Classification: Classifies points into two classes with a neural network using BCEWithLogitsLoss.
Multi-Class Classification: Handles multiple classes with neural networks using CrossEntropyLoss.
Visualization: Plots data distributions and decision boundaries for classification tasks.
Model Persistence: Saves and loads model weights for the linear regression model.
GPU Support: Automatically uses CUDA if available for faster training.

**Table of Contents**

Installation
Usage
Project Structure
Dependencies
Contributing
License

**Installation**
To set up the project locally, follow these steps:

**Clone the repository:**
git clone https://github.com/anwarbarix427/pytorch_model_classification.git
cd pytorch_model_classification


**Create a virtual environment (recommended):**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


**Install dependencies:If a requirements.txt file is provided:**
pip install -r requirements.txt

**Otherwise, install manually:**
pip install torch scikit-learn pandas matplotlib numpy


**Verify setup:Ensure Python 3.7+ is installed. Test by running:**
python linear_regression_model.py



**Usage**
The repository includes five Python scripts, each serving a specific purpose. Below are instructions to run each one:
1. Linear Regression (linear_regression_model.py)

Purpose: Trains a linear regression model on synthetic data to learn weight and bias parameters.
Run:python linear_regression_model.py


Output: Prints the learned weight and bias, compares them to true values, and saves the model to models/linear_regression_model.pth.

2. Binary Classification (binary_classification.py)

Purpose: Trains a neural network to classify points from the make_circles dataset into two classes.
Run:python binary_classification.py


Output: Displays a scatter plot of the dataset and prints training loss and accuracy every 10 epochs.

3. Multi-Class Classification - Blobs (multi_class_model1.py)

Purpose: Trains a neural network to classify points from the make_blobs dataset into four classes.
Run:python multi_class_model1.py


Output: Shows a scatter plot of the dataset and prints training and test loss/accuracy every 10 epochs.

4. Multi-Class Classification - Moons (multi_class_model2.py)

Purpose: Trains a deeper neural network with ReLU activations to classify points from the make_moons dataset into two classes.
Run:python multi_class_model2.py


Output: Plots decision boundaries for training and test sets (using helper_functions.py) and prints training/test loss and accuracy every 10 epochs.

5. Helper Functions (helper_functions.py)

Purpose: Provides utility functions for plotting decision boundaries, calculating accuracy, and other tasks.
Note: This script is not run standalone but is imported by multi_class_model2.py for visualization.

**Notes**

GPU Support: Scripts automatically use CUDA if a GPU is available (device = 'cuda').
Data: All datasets are synthetic, generated using scikit-learn (make_circles, make_blobs, make_moons), so no external data is required.
Model Storage: The linear regression model saves weights to the models/ directory.

**Project Structure**
pytorch_model_classification/
├── linear_regression_model.py     # Linear regression model
├── binary_classification.py       # Binary classification with make_circles
├── multi_class_model1.py          # Multi-class classification with make_blobs
├── multi_class_model2.py          # Multi-class classification with make_moons
├── helper_functions.py            # Utility functions for plotting and metrics
├── models/                        # Directory for saved model weights
│   └── linear_regression_model.pth # Saved linear regression model
├── requirements.txt               # Python dependencies (if provided)
└── README.md                     # Project documentation

**Dependencies**
The project requires the following Python packages:

torch - For building and training neural networks
scikit-learn - For generating synthetic datasets
pandas - For data manipulation in binary_classification.py
matplotlib - For data visualization and decision boundary plotting
numpy - For numerical operations

**Install them manually if requirements.txt is unavailable:**
pip install torch scikit-learn pandas matplotlib numpy

