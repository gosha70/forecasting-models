# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod

END_EVENT_SEQUENCE = 'END'

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
    def train(self, X, y):
        """
        Trains and evaluated the specified model with the dataset of 'X' and the dependent variable/s 'y'.

        Parameters:
        - X: the dataset of independent variables which will be split for training and testing
        - y: the dependent variable/s 'y' 

        Returns:
        - Loss and accuracy for the trained module evaluated on the test dataset
        """    

    @abstractmethod 
    def train_duration_sequence(
            self, 
            event_sequences,
            duration_sequences,
            unique_events_count,
            additional_features, 
            is_random_seq: bool):
        """
        Creates and trains a model with to predict the event duration.

        Parameters:
        - event_sequences
        - duration_sequences
        - unique_events_count
        - additional_features: Does not support yet 
        - is_random_seq: if true, the sequence will randomly selected with a subset of original one

        Returns:
        - Model, loss and accuracy for that trained module
        """           

    @abstractmethod 
    def train_event_sequence(
            self, 
            event_sequences, 
            unique_events_count,
            additional_features, 
            is_random_seq: bool):   
        """
        Creates and trains a  model with to predict time series dataset.

        Parameters:
        - event_sequences
        - unique_events_count
        - additional_features: Does not support yet 
        - is_random_seq: if true, the sequence will randomly selected with a subset of original one

        Returns:
        - Model, loss and accuracy for that trained module
        """    

    def train_remaining_duration(
            self,
            event_sequences,
            duration_sequences,
            unique_events_count,
            y):
        """
        Creates and trains a model to predict remaining case duration from event prefixes.

        Parameters:
        - event_sequences: list of event index sequences (variable length prefixes)
        - duration_sequences: list of cumulative duration sequences matching event prefixes
        - unique_events_count: number of unique events
        - y: target remaining durations (float array)

        Returns:
        - Loss and evaluation metric for the trained model
        """
        raise NotImplementedError(f'{type(self).__name__} does not support train_remaining_duration')

    @abstractmethod
    def predict(self, X):
        """
        Predicts the specified X.

        Parameters:
        - X: the dataset of independent variables which will be used to predict

        Returns:
        - The prediction results
        """

    @abstractmethod
    def predict_proba(self, X):
        """
        Predicts probability distribution over all possible next events.

        Parameters:
        - X: the sequence of events observed so far

        Returns:
        - dict mapping event index (int) to probability (float), summing to 1.0
        """

    @abstractmethod
    def predict_duration(self, X):
        """
        Predicts the duration for the specified X.

        Parameters:
        - X: the dataset of independent variables which will be used to predict

        Returns:
        - The duration prediction
        """ 