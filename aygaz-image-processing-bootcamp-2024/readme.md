# Animal Species Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify 10 different animal species under varying lighting conditions. The model is evaluated on original, manipulated, and color-corrected images to assess its robustness to lighting variations.

## Dataset
The project uses the Animals with Attributes 2 dataset, which can be found on [Kaggle](https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2/data).

### Animal Classes
- Collie
- Dolphin
- Elephant
- Fox
- Moose
- Rabbit
- Sheep
- Squirrel
- Giant Panda
- Polar Bear

## Features
- Image preprocessing and data augmentation
- CNN model implementation using TensorFlow/Keras
- Three different test scenarios:
  1. Original images
  2. Contrast-manipulated images
  3. Color-corrected images using Gray World algorithm

## Requirements
python
numpy
opencv-python
scikit-learn
tensorflow
matplotlib

## Project Structure
- Data preprocessing and loading
- Model architecture:
  - 2 Convolutional layers
  - 2 MaxPooling layers
  - Flatten layer
  - Dense output layer
- Image manipulation techniques:
  - Contrast adjustment
  - Gray World color constancy algorithm

## Results
Current model performance:
- Original Test Set: 52.21% accuracy
- Manipulated Test Set: 46.31% accuracy
- Color-Corrected Test Set: 8.15% accuracy

## Future Improvements
- Implement transfer learning using pre-trained models
- Enhance data augmentation techniques
- Experiment with different color constancy algorithms
- Increase model complexity
- Expand the dataset size
