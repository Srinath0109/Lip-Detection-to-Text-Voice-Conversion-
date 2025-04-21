import numpy as np

class LipLandmarkExtractor:
    def __init__(self):
        self.outer_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        self.inner_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
        self.lip_indices = self.outer_lip + self.inner_lip

    def extract_landmarks(self, face_landmarks):
        try:
            landmarks = []
            
            for idx in self.lip_indices:
                landmark = face_landmarks.landmark[idx]
                x = landmark.x - face_landmarks.landmark[0].x
                y = landmark.y - face_landmarks.landmark[0].y
                z = landmark.z - face_landmarks.landmark[0].z
                landmarks.extend([x, y, z])

            outer_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] 
                                   for i in self.outer_lip])
            inner_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] 
                                   for i in self.inner_lip])

            top_point = np.min(outer_points[:, 1])
            bottom_point = np.max(outer_points[:, 1])
            mouth_height = bottom_point - top_point

            left_point = np.min(outer_points[:, 0])
            right_point = np.max(outer_points[:, 0])
            mouth_width = right_point - left_point

            inner_area = np.abs(np.max(inner_points[:, 0]) - np.min(inner_points[:, 0])) * \
                        np.abs(np.max(inner_points[:, 1]) - np.min(inner_points[:, 1]))

            landmarks.extend([mouth_height, mouth_width, inner_area])

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return np.zeros(len(self.lip_indices) * 3 + 3, dtype=np.float32)
