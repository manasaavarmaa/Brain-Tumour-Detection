import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Define the image directory
image_directory = 'datasets/'

# Define tumor types and their labels
tumor_types = {
    "no_tumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3
}

dataset = []
label = []
INPUT_SIZE = 64

# Loop through the directories for each tumor type
for tumor_type in tumor_types.keys():
    tumor_images = os.listdir(image_directory + tumor_type + '/')
    for image_name in tumor_images:
        if image_name.endswith('.jpg'):
            image = cv2.imread(image_directory + tumor_type + '/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(tumor_types[tumor_type])

# Convert the dataset and label into numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# One-hot encode the labels
num_classes = 4  # Four classes: No Tumor, Glioma, Meningioma, Pituitary
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer for multi-class classification
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, 
          batch_size=16, 
          verbose=1, epochs=10, 
          validation_data=(x_test, y_test), 
          shuffle=False)

# Save the model
model.save('BrainTumor10EpochsMultiClass.h5')

# Load the trained model
model = load_model('BrainTumor10EpochsMultiClass.h5')

# Predicting tumor type from a new image
image = cv2.imread('D:\\Deep Learning Project\\Brain Tumor Image Classification\\pred\\pred0.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

# Expand dimensions to match the model input format
input_img = np.expand_dims(img, axis=0)

# Predict the tumor type
result = model.predict(input_img)
predicted_class = np.argmax(result, axis=1)

# Mapping the predicted class to tumor type
tumor_type_map = {
    0: "No Tumor",
    1: "Glioma",
    2: "Meningioma",
    3: "Pituitary Tumor"
}

print(f"Predicted Tumor Type: {tumor_type_map[predicted_class[0]]}")