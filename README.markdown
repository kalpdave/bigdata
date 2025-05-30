# Big Data: Main Coursework

#### Student Numbers: up2014075, up2161618

### Introduction

CIFAR-100 is a dataset that consists of over 60,000 images that can be distributed across 100 different classifications. This dataset is recognized for its use in benchmarking image classification models. The model created throughout this coursework is a Convolutional Neural Network (CNN) model. This report will describe the basic CNN model architecture needed as well as the convolution layers, pooling strategies, and fully connected layers to identify images. The CNN model will then undergo improvements to increase the accuracy in identifying images.

### Business Objectives
The purpose of the project is to create and evaluate a CNN model that is capable of recognizing images across different categories within the CIFAR-100 dataset by:

- Achieving highly accurate results when classifying images into the 100 distinct classes
- Ensuring the model functions and produces results when tested with unseen data
- Improving and evaluating the CNN architecture to improve efficiency without compromising accuracy

## First ML Pipeline Model

To understand how the model works, a basic model that at least functions had to be established first.

### Overall Code

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from tensorflow.keras.datasets import cifar100
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

# Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Checking data and array shape 
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Splitting training data into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/
def show_samples(data, labels):
    plt.subplots(figsize=(10, 10))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        k = np.random.randint(0, data.shape[0])
        plt.title(int(labels[k]))
        plt.imshow(data[k])
    plt.tight_layout()
    plt.show()

show_samples(x_train, y_train)

# Processing data 
# Converting pixels to float type
# https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# One hot encoding to target classes 
classes = 100
ytrain_categories = to_categorical(y_train, num_classes=100)
yval_categories = to_categorical(y_val, num_classes=100)
ytest_categories = to_categorical(y_test, num_classes=100)

# Building CNN model 
# Uses layers as a 'filtering' system that making model learn based on patterns from training
# https://github.com/uzairlol/CIFAR100-Image-Classification-CNN/blob/main/Item%20Image%20Model%20Training%20and%20Evaluation.ipynb
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
])

model.summary()

# Beginning the training of model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Main training section
# ChatGPT helped to structure how model can undergo training
history = model.fit(x_train, ytrain_categories, epochs=25, batch_size=64, validation_data=(x_val, yval_categories))

test_loss, test_accuracy = model.evaluate(x_test, ytest_categories)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualisation of results through graphs using matplotlib
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
```

### 1. Data Collection and Processing
The CIFAR-100 dataset was sourced from the TensorFlow (Keras) library using `cifar100.load_data()` from `tensorflow.keras.datasets`. TensorFlow was a smoother option compared to unpickling data from other sources.

### 2. Exploratory Data Analysis (EDA)
```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

show_samples(x_train, y_train)
```
This section prints and displays the shape of the training and test datasets to confirm dimensions. It also shows a random sample of images from the dataset using Matplotlib.

### 3. CNN Model
To filter data and patterns from the images, a layered architecture was implemented:

- **Convolutional Layering (`Conv2D`)**: Extracts features from images by applying filters.
- **Pooling Layers (`MaxPooling2D`)**: Reduces spatial dimensions in feature maps.
- **Flattening Layers (`Flatten`)**: Transforms 2D feature maps into 1D vectors.
- **Dense Layers**: Connects layers for classification.
- **Dropout Layer**: Prevents overfitting by randomly setting nodes to zero during training.

The model was compiled using:
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- **Optimizer (`Adam`)**: Dynamically adjusts parameter updates for faster convergence.
- **Loss Function (`categorical_crossentropy`)**: Quantifies the difference between predicted and actual distributions.
- **Metric (`accuracy`)**: Measures the fraction of correct predictions.

### 4. Training Model
```python
history = model.fit(x_train, ytrain_categories, epochs=25, batch_size=64, validation_data=(x_val, yval_categories))
```
Training was conducted over 25 epochs with batch processing for efficient computation, monitoring training and validation accuracy and loss per epoch.

### 5. Prediction, Results, and Evaluation
```python
test_loss, test_accuracy = model.evaluate(x_test, ytest_categories)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
```
The model was tested with unseen test data, achieving a base accuracy of 36.9%. Adjusting the training-validation split (e.g., 85-15 or 95-5) resulted in accuracies of 33.9% and 36.6%, respectively. Increasing the width of convolutional layers and dense layers improved accuracy to around 39%:
```python
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
])
```
Increasing epochs to 50 or 100 improved accuracy by 1-2%, stabilizing performance but not significantly amplifying it.

## Second ML Pipeline Model

This model was developed based on an example provided by the university lecturer, refined with assistance from AI tools like DeepSeek and ChatGPT to polish errors.

### Overall Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
num_classes = len(np.unique(y_train))

# Data quality check
def check_pics(data, dataset_name):
    bad_imgs = 0
    good_imgs = 0
    for i, img in enumerate(data):
        if not isinstance(img, np.ndarray):
            print(f"{dataset_name} img {i}: Not an array")
            bad_imgs += 1
            continue
        if img.shape != (32, 32, 3):
            print(f"{dataset_name} img {i}: Shape {img.shape}, need (32, 32, 3)")
            bad_imgs += 1
            continue
        if not (img.dtype == np.uint8 and img.min() >= 0 and img.max() <= 255):
            print(f"{dataset_name} img {i}: Bad pixels, min={img.min()}, max={img.max()}")
            bad_imgs += 1
            continue
        if np.isnan(img).any():
            print(f"{dataset_name} img {i}: NaN found")
            bad_imgs += 1
            continue
        good_imgs += 1
    print(f"{dataset_name}: {good_imgs} good, {bad_imgs} bad")

check_pics(X_train, "Train")
check_pics(X_test, "Test")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=15)

# Normalize and encode
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Lightweight CNN
def light_cnn(shape, classes):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=shape, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn1 = light_cnn((32, 32, 3), num_classes)
cnn1.summary()

# Training
stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128).prefetch(tf.data.AUTOTUNE)

cnn1_hist = cnn1.fit(train_data, epochs=20, validation_data=val_data, callbacks=[stop])

# Evaluation
cnn1_loss, cnn1_acc = cnn1.evaluate(X_test, y_test)
print(f"Light CNN: Loss = {cnn1_loss:.2f}, Acc = {cnn1_acc * 100:.2f}%")
```

### 1. Data Collection and Processing
The CIFAR-100 dataset was loaded using `cifar100.load_data()` from TensorFlow. Data was normalized by converting pixel values to floats and scaling to [0,1]. Labels were one-hot encoded for multi-class classification.

### 2. EDA
A data quality check ensured images were valid NumPy arrays with shape (32, 32, 3), pixel values in [0, 255], and no NaNs. The dataset was split into training (80%) and validation (20%) sets. Class distribution was visualized using bar plots to confirm balanced classes.

### 3. CNN Model
A lightweight CNN was designed for faster training:
- **Conv2D Layers**: 16 and 32 filters with 3x3 kernels, ReLU activation, and same padding.
- **MaxPooling2D**: 2x2 pooling to reduce spatial dimensions.
- **Flatten and Dense Layers**: 128-unit ReLU layer followed by a 100-unit softmax layer for classification.
No dropout was included, potentially affecting generalization. The model was compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metrics.

### 4. Training
Training used `tf.data` for efficient data loading, with a batch size of 128 and 20 epochs. Early stopping monitored validation loss with a patience of 5 to prevent overfitting.

### 5. Prediction, Results, and Evaluation
The model was evaluated on the test set, achieving an accuracy of approximately 30-35%. A confusion matrix and classification report provided insights into class-wise performance. Single-image predictions and top-10 probability visualizations were generated to assess model behavior.

## Third ML Pipeline Model
This model combines improvements from the first and second models, incorporating data augmentation and a deeper CNN architecture to achieve higher accuracy.

### Overall Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
num_classes = len(np.unique(y_train))

# Data quality check
def check_pics(data, dataset_name):
    bad_imgs = 0
    good_imgs = 0
    for i, img in enumerate(data):
        if not isinstance(img, np.ndarray):
            print(f"{dataset_name} img {i}: Not an array")
            bad_imgs += 1
            continue
        if img.shape != (32, 32, 3):
            print(f"{dataset_name} img {i}: Shape {img.shape}, need (32, 32, 3)")
            bad_imgs += 1
            continue
        if not (img.dtype == np.uint8 and img.min() >= 0 and img.max() <= 255):
            print(f"{dataset_name} img {i}: Bad pixels, min={img.min()}, max={img.max()}")
            bad_imgs += 1
            continue
        if np.isnan(img).any():
            print(f"{dataset_name} img {i}: NaN found")
            bad_imgs += 1
            continue
        good_imgs += 1
    print(f"{dataset_name}: {good_imgs} good images, {bad_imgs} bad images")

check_pics(X_train, "Train")
check_pics(X_test, "Test")

# Data preparation
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# Data splitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=15)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# CNN model
cnn1 = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')
])

cnn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
cnn_training = cnn1.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[stop]
)

# Evaluation
test_loss, test_accuracy = cnn1.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
```

### 1. Data Collection and Processing
The CIFAR-100 dataset was loaded from TensorFlow. Images were normalized to [0,1], and labels were one-hot encoded. A data quality check validated image arrays, shapes, and pixel values.

### 2. EDA
The training data was split into 80% training and 20% validation sets. Class distribution was analyzed to ensure balance, similar to the second model.

### 3. CNN Model
A deeper CNN was designed:
- **Conv2D Layers**: Three layers with 128 filters each, 3x3 kernels, and ReLU activation.
- **MaxPooling2D**: 2x2 pooling after each convolutional layer.
- **Flatten and Dense Layers**: 256-unit ReLU layer with 0.5 dropout, followed by a 100-unit softmax layer.
The model was compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metrics.

### 4. Training
Data augmentation was applied (rotation, shifts, flips, zoom) to enhance training and reduce overfitting. The model was trained for 50 epochs with a batch size of 64, using early stopping to monitor validation loss.

### 5. Prediction, Results, and Evaluation
The model achieved a test accuracy of approximately 40-45%. Multi-image visualizations showed predictions with true labels and correctness. Single-image predictions included top-10 probability plots. A confusion matrix and classification report detailed class-wise performance.

## Future Work
With a working model, the primary objectives for future work would be to improve the processing time taken when training the model for each epoch as well as increasing the accuracy.

Processing time could be cut by introducing batch normalization, which normalizes layer inputs, making training faster and more stable.

To increase accuracy, image augmentation can be further tuned, or advanced architectures like ResNet or transfer learning with pre-trained models (e.g., MobileNetV2) could be explored.

## Libraries and Modules
### Libraries
- **NumPy**: Used for numerical operations and array management.
- **Pandas**: Facilitated data frame operations for class counts and results.
- **Matplotlib**: Used for visualization and plotting data and results.
- **Seaborn**: Enhanced visualization for bar plots and confusion matrices.
- **TensorFlow**: The main library for machine learning, providing the CIFAR-100 dataset, network creation, compilation, and training.
- **Keras**: Built on TensorFlow, used for defining and training neural networks, providing dataset access, CNN layers, and utilities like `to_categorical`.
- **Scikit-learn**: Provided tools for data splitting and model evaluation.

### Modules
- **keras.datasets**: Accesses datasets like CIFAR-100.
- **keras.utils**: Provides utilities like one-hot encoding.
- **keras.layers**: Defines neural network layers.
- **keras.models**: Manages neural network models.
- **tensorflow.keras.datasets**: Provides dataset loaders.
- **keras.preprocessing.image**: Supports data augmentation.

## Unresolved Issues and Bugs
Most issues occurred during code construction, such as incorrect labels in functions. Kernel crashes required refreshing before running code, possibly due to internet connection issues or CPU memory constraints.

## Conclusions
Over this project, a model has successfully been created that predicts and identifies images from the CIFAR-100 dataset with up to 45% accuracy. The progression from a basic CNN (36.9%) to a lightweight model (30-35%) and a deeper, augmented model (40-45%) demonstrates improvements in architecture and training strategies. The project provided insights into ML pipelines, CNN architectures, and their application in image classification.

## References and Acknowledgments
- [GeeksforGeeks: Image Classification using CIFAR-100](https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/): Structured EDA for the first model.
- [GitHub: LeoTungAnh/CNN-CIFAR-100](https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb): Guided data processing and one-hot encoding for the first model.
- [GitHub: uzairlol/CIFAR100-Image-Classification-CNN](https://github.com/uzairlol/CIFAR100-Image-Classification-CNN/blob/main/Item%20Image%20Model%20Training%20and%20Evaluation.ipynb): Inspired CNN architecture for the first model.
- [StackExchange: Epochs in CNN](https://datascience.stackexchange.com/questions/46523/is-a-large-number-of-epochs-good-or-bad-idea-in-cnn): Informed epoch selection for the third model.
- University lecturer’s coursework support file: Guided the second model’s structure.
- AI tools (ChatGPT, DeepSeek, Grok): Provided limited assistance in debugging, structuring code, and refining model configurations within permitted coursework guidelines.

## Terms and Definitions
- *Overfitting*: When a model learns training data too well, including noise, leading to poor generalization.
- *Optimizer*: Algorithm to minimize loss by updating model parameters.
- *Epoch*: A complete pass through the training dataset.
- *One-Hot Encoding*: Converts categorical variables into binary vectors.
- *Categorical Cross-Entropy*: Loss function for multi-class classification, measuring discrepancy between predicted and true distributions.
- *Convergence*: When the optimization algorithm stabilizes with minimal loss changes.
- *Augmentation*: Modifies data through transformations to enhance training.
- *Normalization*: Converts data to a standard format for easier analysis.