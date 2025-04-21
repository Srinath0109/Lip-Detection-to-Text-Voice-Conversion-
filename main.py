import cv2
import mediapipe as mp
import numpy as np
from utils.landmark_extractor import LipLandmarkExtractor
from utils.sequence_processor import SequenceProcessor
from models.lip_reader import LipReader
from models.text_processor import TextProcessor
from utils.text_to_speech import TextToSpeechEngine

class LipReadingSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access webcam")
        
        self.lip_extractor = LipLandmarkExtractor()
        self.sequence_processor = SequenceProcessor(sequence_length=30)
        self.lip_reader = LipReader()
        self.text_processor = TextProcessor()
        self.tts_engine = TextToSpeechEngine()
        self.training_mode = False
        self.current_training_word = None
        self.training_count = {}
        self.target_samples = 5
        
        for word in self.lip_reader.vocabulary:
            self.training_count[word] = 0
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def run(self):
        current_text = ""
        print("\nPress 't' to enter training mode")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                lip_landmarks = self.lip_extractor.extract_landmarks(
                    results.multi_face_landmarks[0])
                
                sequence_ready = self.sequence_processor.add_frame(lip_landmarks)
                
                if sequence_ready:
                    sequence = self.sequence_processor.get_current_sequence()
                    
                    if self.training_mode and self.current_training_word:
                        self.lip_reader.train(sequence, self.current_training_word)
                        self.training_count[self.current_training_word] += 1
                        print(f"\nTrained '{self.current_training_word}' - {self.training_count[self.current_training_word]}/{self.target_samples} samples")
                        self.training_mode = False
                        self.current_training_word = None
                        current_text = "Training complete"
                    else:
                        predicted_text = self.lip_reader.predict(sequence)
                        if predicted_text:
                            current_text = predicted_text
                            self.tts_engine.speak(predicted_text)

            # UI Elements
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            
            mode_text = "TRAINING MODE" if self.training_mode else "PREDICTION MODE"
            cv2.putText(frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Current: {current_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Training status
            y_pos = 100
            cv2.putText(frame, "Training Progress:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            for word in self.lip_reader.vocabulary:
                y_pos += 20
                status = f"{word}: {self.training_count[word]}/{self.target_samples}"
                cv2.putText(frame, status, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx in self.lip_extractor.lip_indices:
                        pt = face_landmarks.landmark[idx]
                        x = int(pt.x * frame.shape[1])
                        y = int(pt.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow('Lip Reading System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.training_mode = True
                print("\nAvailable words to train:")
                for word in self.lip_reader.vocabulary:
                    print(f"- {word} ({self.training_count[word]}/{self.target_samples} samples)")
                word = input("\nEnter word to train: ")
                if word in self.lip_reader.vocabulary:
                    self.current_training_word = word
                    current_text = f"Say '{word}' clearly"
                else:
                    print("Invalid word")
                    self.training_mode = False

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == "__main__":
    system = LipReadingSystem()
    system.run()
