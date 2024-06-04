# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

from .base_model_factory import BaseModelFactory

class MarkivChainModelFactory(BaseModelFactory):
    def __init__(self):
        super().__init__()
        self.transition_matrix = None

    def create_classification_model(self, **kwargs):
        raise ValueError('MarkivChainModelFactory', 'create_classification_model()')
    
    def create_regression_model(self, **kwargs):    
        raise ValueError('MarkivChainModelFactory', 'create_regression_model()')
    
    def train(self, X, y):
        raise ValueError('MarkivChainModelFactory', 'train()')
    
    def train_duration_sequence(
            self, 
            event_sequences,
            duration_sequences,
            unique_events_count,
            additional_features, 
            is_random_seq: bool):
        raise ValueError('MarkivChainModelFactory', 'train_duration_sequence()')
    
    def predict_duration(self, X):   
        raise ValueError('MarkivChainModelFactory', 'predict_duration()')
    
    def generate_simple_event_sequence(self, event_sequences):
        # Prepare X and y datasets
        X = []
        y = []
        for _, seq in enumerate(event_sequences):
            len_seq = len(seq)
            if len_seq < 2:
                continue

            # Generate a random length for the subsequence
            random_length = random.randint(1, len_seq - 1)
            combined_seq = seq[:random_length]

            # Define the target value
            x_y = seq[random_length] if random_length < len(seq) - 1 else self.unique_events_count

            X.append(combined_seq)
            y.append(x_y)
            #print(f"X: {combined_seq} -> y: {x_y}")
        return X, y
    
    def evaluate(self, X, y):
        correct_predictions = 0
        total_predictions = len(X)

        log_likelihood = 0

        for seq, true_next in zip(X, y):
            predicted_next = self.predict(seq)
            if predicted_next == true_next:
                correct_predictions += 1
            
            # Calculate the log likelihood
            current_event = seq[-1]
            next_event_prob = self.transition_matrix.get(current_event, {}).get(true_next, 1e-10)  # Avoid log(0)
            log_likelihood += np.log(next_event_prob)

        accuracy = correct_predictions / total_predictions
        loss = -log_likelihood / total_predictions

        return loss, accuracy
    
    def train_event_sequence(
            self, 
            event_sequences,
            unique_events_count,
            additional_features, 
            is_random_seq: bool):          
        self.unique_events_count = unique_events_count 
        transition_counts = defaultdict(lambda: defaultdict(int))

        for seq in event_sequences:
            for i in range(len(seq) - 1):
                transition_counts[seq[i]][seq[i + 1]] += 1

        self.transition_matrix = {
            state: {next_state: count / sum(next_states.values())
                    for next_state, count in next_states.items()}
            for state, next_states in transition_counts.items()
        }
        print(f"Transition matrix: {self.transition_matrix}")

        X, y = self.generate_simple_event_sequence(event_sequences)

        # Evaluate the Markov Chain model
        loss, accuracy = self.evaluate(X, y)

        return loss, accuracy
       
    def predict(self, X):
        current_event = X[-1]
        next_events = self.transition_matrix.get(current_event, {})
        if not next_events:
            return len(self.transition_matrix)  # End event sequence
        next_event = max(next_events, key=next_events.get)
        return next_event
    