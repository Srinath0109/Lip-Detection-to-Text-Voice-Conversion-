import numpy as np
import json
import os

class LipReader:
    def __init__(self):
        self.vocabulary = ['hello', 'yes', 'no', 'thank you', 'please']
        self.last_prediction = None
        self.cooldown = 0
        self.frame_buffer = []
        self.buffer_size = 15
        self.training_data = {}
        self.load_training_data()

    def load_training_data(self):
        try:
            if os.path.exists('lip_patterns.json'):
                with open('lip_patterns.json', 'r') as f:
                    self.training_data = json.load(f)
        except Exception as e:
            print(f"Error loading training data: {e}")
            self.training_data = {}

    def save_training_data(self):
        with open('lip_patterns.json', 'w') as f:
            json.dump(self.training_data, f)

    def train(self, sequence, word):
        if word not in self.vocabulary:
            return

        if word not in self.training_data:
            self.training_data[word] = []

        pattern = {
            'height': float(np.mean([frame[-3] for frame in sequence])),
            'width': float(np.mean([frame[-2] for frame in sequence])),
            'area': float(np.mean([frame[-1] for frame in sequence]))
        }
        self.training_data[word].append(pattern)
        self.save_training_data()

    def predict(self, sequence):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if not self.training_data:
            return None

        try:
            current_pattern = {
                'height': float(np.mean([frame[-3] for frame in sequence])),
                'width': float(np.mean([frame[-2] for frame in sequence])),
                'area': float(np.mean([frame[-1] for frame in sequence]))
            }

            best_match = None
            best_score = float('inf')

            for word, patterns in self.training_data.items():
                for pattern in patterns:
                    score = sum((current_pattern[k] - pattern[k])**2 
                              for k in ['height', 'width', 'area'])
                    if score < best_score:
                        best_score = score
                        best_match = word

            if best_match and best_score < 0.01:
                if best_match != self.last_prediction:
                    self.last_prediction = best_match
                    self.cooldown = 10
                    return best_match

        except Exception as e:
            print(f"Error in prediction: {e}")
            
        return None
