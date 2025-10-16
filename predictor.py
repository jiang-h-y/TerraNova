"""
predictor.py
Predictor class for neural network
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

class Predictor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_shape = len(X.columns)
        self.output_classes = len(y.columns)
    
    def fit_model(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size
            )
        
        early_stopping = EarlyStopping(
            min_delta=0.001,
            patience=5,
            restore_best_weights=True
            )

        self.model = keras.Sequential([
            layers.Input(shape=(self.input_shape,)),
            layers.Dense(units=32, activation="relu"),
            layers.Dense(units=16, activation="relu"),
            layers.Dense(units=self.output_classes, activation="softmax")
            ])
        
        self.model.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split = 0.2,
            epochs=50,
            callbacks=[early_stopping])
        
        loss, acc = self.model.evaluate(X_test, y_test)
        print(f"Accuracy: {acc * 100:.2f}%")

    def plot_loss(self):
        history_df = pd.DataFrame(self.history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()