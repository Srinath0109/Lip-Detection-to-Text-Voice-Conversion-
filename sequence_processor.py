import numpy as np

class SequenceProcessor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.current_sequence = []

    def add_frame(self, landmarks):
        self.current_sequence.append(landmarks)
        
        if len(self.current_sequence) > self.sequence_length:
            self.current_sequence.pop(0)
            
        return len(self.current_sequence) == self.sequence_length

    def get_current_sequence(self):
        return np.array(self.current_sequence)
