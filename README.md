# CNN-LSTM
## Human Action Recognition Using CNN-LSTM
This project implements a hybrid CNN-LSTM model for human action recognition using the UCF50 dataset. The model processes video frames to classify different activities such as WalkingWithDog, TaiChi, Swing, and HorseRace.

# 📌 Project Overview
Utilizes ConvLSTM2D layers to capture both spatial and temporal patterns in videos.
Extracts frames from videos, resizes and normalizes them for model training.
Implements transfer learning with MobileNetV2 for efficient feature extraction.
Trains on a selected subset of UCF50 dataset and supports easy extension to more classes.
Includes Grad-CAM visualization to interpret model predictions.
# 📂 Dataset
Source: UCF50 Human Action Dataset
Classes Used: WalkingWithDog, TaiChi, Swing, HorseRace (extendable)
Input Shape: (20, 64, 64, 3) (20 frames per video, resized to 64x64)
# 🛠 Model Architecture
The model follows a CNN-LSTM hybrid approach:

ConvLSTM2D Layers: Extract spatiotemporal features from video frames.
MaxPooling3D: Reduces spatial dimensions for better generalization.
Dropout Layers: Prevent overfitting.
Fully Connected (Dense) Layer: Outputs probability distribution over action classes.

# 📝 Training Details
Loss Function: Categorical Crossentropy
Optimizer: Adam
Batch Size: 4
Validation Split: 20%
Early Stopping: Enabled (patience = 18 epochs)
# 📊 Model Evaluation
Trained model is evaluated using a confusion matrix and accuracy metrics.

# 🔍 Visualizations
Confusion Matrix: Evaluates classification performance.
Loss & Accuracy Curves: Helps track training progress.

