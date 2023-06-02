Lung Illness Diagnosis using X-ray Scans

Overview

This repository contains the code and resources for developing a learning model that supports doctors in diagnosing illnesses affecting patients' lungs based on X-ray scans. The goal is to create a classification model using the Keras module that can accurately predict the presence of pneumonia, Covid-19, or no illness based on an input X-ray scan.

Dataset

The dataset used for training and evaluating the model consists of a collection of X-ray lung scans. Each scan is labeled with the corresponding diagnosis: pneumonia, Covid-19, or no illness. The dataset is carefully curated and annotated to ensure accuracy and reliability.

Model Development

The model is developed using the Keras module, a powerful deep learning library in Python. The model architecture consists of multiple layers, including input layers, hidden layers, and output layers. Various techniques such as convolutional neural networks (CNNs) and dense layers are employed to effectively capture the features in the X-ray scans and make accurate predictions.

Usage

To use the model for lung illness diagnosis, follow these steps:

1. Install the required dependencies by running pip install -r requirements.txt.
2. Preprocess the dataset and prepare the X-ray scans for training.
3. Run the training script train.py to train the model on the dataset.
4. After training, the model will be saved to a file for future use.
5. Use the trained model to predict the diagnosis for new X-ray scans using the predict.py script.

Evaluation

The model's performance is evaluated using various metrics, including accuracy, precision, recall, and F1 score. Extensive testing is conducted on a separate test set to ensure the model's generalization capabilities and accuracy in real-world scenarios.
