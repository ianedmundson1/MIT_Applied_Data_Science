# Facial Emotion Detection Project  

## Overview  
This project focuses on building a deep learning model to detect facial emotions from images. The model categorizes emotions into four classes: `happy`, `sad`, `neutral`, and `surprise`. The project leverages transfer learning using the VGG16 model and hyperparameter tuning to achieve optimal performance.  

## Problem Definition  
Facial emotions constitute a significant part of human communication. Understanding these emotions can enhance human-computer interaction, improve health diagnostics, and enable intuitive AI systems. The goal of this project is to train a model that accurately classifies human emotions based on facial expressions.  

## Dataset  
The dataset consists of three folders: `train`, `validation`, and `test`. Each folder contains images categorized into four subfolders:  
- **happy**: Images of people with happy expressions.  
- **sad**: Images of people with sad expressions.  
- **neutral**: Images of people with neutral expressions.  
- **surprise**: Images of people with surprised expressions.  

## Methodology  
1. **Data Preprocessing**:  
    - Images are resized to 48x48 pixels and normalized.  
    - Data augmentation techniques like random flipping and rotation are applied.  

2. **Model Architecture**:  
    - The project uses the VGG16 model pre-trained on ImageNet as the base model.  
    - Additional layers are added for fine-tuning, including dense layers, batch normalization, and dropout layers.  

3. **Hyperparameter Tuning**:  
    - The `keras_tuner` library is used to optimize the number of neurons in dense layers.  
    - The `Hyperband` algorithm is employed to find the best configuration.  

4. **Training**:  
    - The model is trained using the Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`).  
    - Early stopping is used to prevent overfitting.  

5. **Evaluation**:  
    - The model's performance is evaluated on the test set using accuracy, confusion matrix, and classification reports.  

## Results  
- The final model achieved a test accuracy of approximately **72%**.  
- The confusion matrix and classification report highlight the model's performance across all emotion classes.  

## Visualizations  
- Training and validation accuracy/loss plots are generated to monitor the model's performance over epochs.  
- A heatmap of the confusion matrix is plotted to visualize the classification results.  

## How to Run  
1. Clone the repository and navigate to the project directory.  
2. Ensure the dataset is downloaded and placed in the `Facial_emotion_images` folder.  
3. Run the `FINAL_SUBMISSION.ipynb` notebook to train and evaluate the model.  

## Dependencies  
- Python 3.10  
- TensorFlow 2.x  
- Keras  
- Keras Tuner  
- Matplotlib  
- Seaborn  
- NumPy  
- PIL  

## Acknowledgments  
- The VGG16 model is pre-trained on the ImageNet dataset.  
- The dataset used for this project was provided via Olympus.  

## Future Work  
- Extend the model to detect additional emotions.  
- Experiment with other transfer learning models like ResNet or EfficientNet.  
- Deploy the model as a web application for real-time emotion detection.  
- Optimize the model for faster inference on edge devices.  
