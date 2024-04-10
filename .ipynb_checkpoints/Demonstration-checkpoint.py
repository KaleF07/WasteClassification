import os
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt
import time
import tempfile
import numpy as np
from fastai.vision.all import Learner

load_model = tf.keras.models.load_model('mobilenet_model.h5', compile=True)

# Convert model from type Functional to Sequential
mobilenet_model = tf.keras.models.Sequential()
for layer in load_model.layers:
  mobilenet_model.add(layer)

import cv2
import numpy as np
import tkinter as tk
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from PIL import Image, ImageTk

# Function to preprocess the image
def preprocess_frame(frame):
    # Resize the frame to match the input size of your model
    frame = cv2.resize(frame, (224, 224))
    # Preprocess the frame for your model
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Function to predict the class of the image
def predict_class(frame):
    # Preprocess the frame
    frame = preprocess_frame(frame)
    # Make predictions
    preds = mobilenet_model.predict(frame)
    # Get the predicted class
    pred_class = "Recyclable" if preds[0][0] > preds[0][1] else "Organic"
    return pred_class

# Function to update the GUI with the webcam feed
def update_frame():
    ret, frame = cap.read()  # Read a frame from the webcam
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
    pred_class = predict_class(frame)  # Predict the class of the image

    # Update the GUI with the current frame and predicted class
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    panel.img = img
    panel.config(image=img)
    label.config(text="Predicted Class: " + pred_class)

    # Schedule the next update after 10 milliseconds
    panel.after(10, update_frame)

# Create the main GUI window
root = tk.Tk()
root.title("Webcam Feed")

# Create a panel to display the webcam feed
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Create a label to display the predicted class
label = tk.Label(root, text="")
label.pack(padx=10, pady=5)

# Open the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Start updating the GUI with the webcam feed
update_frame()

# Run the GUI application
root.mainloop()

# Release the webcam
cap.release()