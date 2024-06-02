# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod

class BaseModelFactory(ABC):
    def __init__(self):
        """Initializes base ML model (algorythm or network)."""

    @abstractmethod
    def create_classification_model(self, **kwargs):
        """
        Creates the classification model.

        Parameters:
        - (**kwargs): the optional hyperparameters specific for this model 

        Returns:
        - ML model
        """
       
    @abstractmethod 
    def create_regression_model(self, **kwargs):
        """
        Creates the regression model.

        Parameters:
        - (**kwargs): the optional hyperparameters specific for this model 

        Returns:
        - ML model
        """
      
    @abstractmethod 
    def train(self, model, X, y):
        """
        Trains and evaluated the specified model with the dataset of 'X' and the dependent variable/s 'y'.

        Parameters:
        - model: the not trained model to be trained
        - X: the dataset of independent variables which will be split for training and testing
        - y: the dependent variable/s 'y' 

        Returns:
        - Loss and accuracy for the trained module evaluated on the test dataset
        """    