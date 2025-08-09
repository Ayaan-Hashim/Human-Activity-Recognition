# Human-Activity-Recognition
A hierarchical machine learning framework comprising two advanced CNN models capable of distinguishing among 11 distinct physical activities and 4 unique breathing patterns. The `tflite` files can be found in `andrioid-app/app/src/main/assets/Models`. Some tasks code can also be found under `model_code/`.


# On-Device Activity Classification System

This project implements a dual-model CNN architecture for classifying physical and breathing activities using sensor data, designed for on-device deployment in mobile applications.

## Overview
- **Hierarchical classification**: Physical activities → Breathing activities (only for stationary states)
- **Input data**: Accelerometer (x,y,z) and Gyroscope (x,y,z) readings
- **Output classes**: 
  - 11 Physical activities
  - 4 Breathing activities (with automatic fallback to "Normal" during movement)

## Data Pipeline
### Preprocessing Steps:
1. **Data Integration**:
   - Concatenated multi-subject data into unified dataframe
   - Added two label columns:
     - Physical Activity (11 classes)
     - Breathing Activity (4 classes + -1 for dynamic activities)

2. **Window Segmentation**:
   - Sliding window size: 50 samples
   - Overlap: 90% (step size: 5)
   - Segment labeling: Mode of contained labels

3. **Label Encoding**:
   - One-hot encoding using `to_categorical()`

## Model Architecture
### Physical Activity Classifier (First Stage)
- **Input**: Sensor data windows (6 channels)
- **Layers**:
  - 3 × Conv1D (64 filters, kernel=3, ReLU, L2 reg)
  - Batch Normalization + Dropout (0.2) after each Conv
  - Flatten layer
  - Dense (128, ReLU)
  - Output (11, softmax)

- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Batch Size: 64
  - Train-Validation Split: 80-20

### Breathing Activity Classifier (Second Stage)
- **Input**: Sensor data windows (only when stationary)
- **Layers**:
  - 3 × Conv1D (64 filters, kernel=3, ReLU)
  - Batch Normalization + Dropout (0.2) after each Conv
  - Flatten layer
  - Dense (128, ReLU)
  - Dense (32, ReLU)
  - Output (4, softmax)

- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Batch Size: 128
  - Train-Validation Split: 80-20

**Inference Logic**:
```mermaid
graph TD
    A[Sensor Data] --> B{Physical Activity Model}
    B -->|Dynamic| C[Set Breathing=Normal]
    B -->|Stationary| D[Breathing Activity Model]
