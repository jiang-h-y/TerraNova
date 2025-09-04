import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from prototype.land import Land

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Neural network class

class Predictor:

    def __init__(self, data): # a list of Lands:
        
        X = np.array([land.to_array()[0] for land in data])
        Y = np.array([land.to_array()[1] for land in data])
        
        number_of_possible_outcomes = 15 # Whatever the number of possible outputs (building types) we have
        
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        
        x_train = X[split_index:]
        y_train = Y[split_index:]
        x_test = X[:split_index]
        y_test = Y[:split_index]

        # Define the neural network model
        self.model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
            keras.layers.Dense(number_of_possible_outcomes, activation='softmax')  # Output layer with 15 neurons (for 15 classes)
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        # Train the model
        self.model.fit(x_train, y_train, epochs=5)

        # Evaluate the model
        self.test_loss, self.test_acc = self.model.evaluate(x_test, y_test)
        print("\nTest accuracy:", self.test_acc)
        
        self.xtest, self.ytest = x_test, y_test


    def makePrediction(self, entry):
        # Ensure entry is a 1D array of length 3
        entry = np.array(entry) 
        # Ensure the input has the correct shape (1, 3) for the model
        if entry.shape == (3,):
            entry = entry.reshape(1, 3)  # Reshape it to (1, 3)
        # Predict probabilities for each class
        prediction = self.model.predict(entry)
        # Get the predicted class label (the class with the highest probability)
        return np.argmax(prediction, axis=1)[0]
        
    def displayPrediction(self):
        array_test = np.array(self.ytest)
        array_predict = np.array([self.makePrediction(x) for x in self.xtest])

        # Calculate the discrepancy
        discrepancy = array_predict - array_test

        # Create a figure with two subplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # First plot: Predicted vs. Test values
        ax[0].plot(array_test, label="Test Values", marker='o', linestyle='', color='blue')
        ax[0].plot(array_predict, label="Predicted Values", marker='x', linestyle='', color='red')
        ax[0].set_title("Predicted vs. Test Values")
        ax[0].set_xlabel("Index")
        ax[0].set_ylabel("Value")
        ax[0].legend()


        # Show the plot
        plt.show()
                
