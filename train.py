import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


###Data Preprocessing###

###Observations:

#What types of images are we working with? Images are x-rays of lungs. These are in grayscale color format
#How are they organized? How will this affect preprocessing? Data split into train and test
#Will this be binary classification or multi-class classification?
#These images are classified in three categories 1)Normal 2)Covid19 3)Pneumonia 

#import datasets
train_data_directory = 'augmented-data/train'
test_data_directory = 'augmented-data/test'


#batch_size and epochs variable for testing
batch = 12
epochs = 5

#create image generator instance and iterators 
data_generator = ImageDataGenerator(rescale = 1/255) 
train_iterator = data_generator.flow_from_directory(train_data_directory, class_mode= 'categorical', color_mode = 'grayscale', batch_size = batch)

test_iterator = data_generator.flow_from_directory(test_data_directory, class_mode= 'categorical', color_mode = 'grayscale', batch_size = batch)

#build the model

#instantiate
model = tf.keras.Sequential()

#input layer, shape based on size of images and number of chanels fmodel = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1))) #Input
model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(3,activation="softmax")) #output

model.summary()
#compile the model with Adam optimizer and crossentropy for the labels due to one-hot encoding, scoring metrics using AUC and Accuracy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])


#Use model.fit(...) to train and validate our model for 5 epochs:

model.fit(
       train_iterator,
       steps_per_epoch=train_iterator.samples/batch,
       epochs=epochs,
       validation_data=test_iterator,
       validation_steps=test_iterator.samples/batch)


#This code calculates the confusion matrix and classification report using the predicted labels (y_pred) and the true labels (y_true). The confusion matrix provides an overview of the model's performance for each class, while the classification report provides metrics such as precision, recall, and F1-score for each class.

# Make predictions on the test set
predictions = model.predict(test_iterator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_iterator.classes

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Create the classification report

# maps class names to their corresponding index values in list format
class_labels = list(test_iterator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)
