# Lip-Detection-to-Text-Voice-Conversion-

# Lip Reading System

A real-time lip reading application that recognizes spoken words through webcam input.
<img width="1440" alt="Screenshot 2025-04-21 at 11 18 05 AM" src="https://github.com/user-attachments/assets/538052e8-9f10-4bb6-a93d-209a75a22eda" />


## Features

- Real-time lip movement detection
- Word recognition for basic vocabulary
- Training mode for personalized accuracy
- Visual feedback with facial landmarks
- Text-to-speech output
- Progress tracking for training samples

  ![image](https://github.com/user-attachments/assets/36ff5498-7632-455d-89db-4ed0c2f49ba3)


## Prerequisites

- Python 3.7+
- Webcam access
- macOS (for text-to-speech functionality)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lip_reading_system

## Usage
1. Run the application:
```bash
python3 src/main.py
 ```

2. Training Mode:
   
   - Press 't' to enter training mode
   - Enter a word from the vocabulary when prompted
   - Say the word clearly while facing the camera
   - Repeat for each word multiple times
3. Prediction Mode:
   
   - After training, the system automatically predicts spoken words
   - Visual feedback shows detected lip movements
   - Predictions are displayed and spoken through text-to-speech
4. Exit:
   
   - Press 'q' to quit the application
## Vocabulary
Current supported words:

- hello
- yes
- no
- thank you
- please
## Project Structure
```plaintext
lip_reading_system/
├── src/
│   ├── main.py                 # Main application
│   ├── models/
│   │   ├── lip_reader.py       # Lip reading logic
│   │   └── text_processor.py   # Text processing
│   └── utils/
│       ├── landmark_extractor.py    # Facial landmark detection
│       ├── sequence_processor.py    # Frame sequence handling
│       └── text_to_speech.py        # Speech output
 ```
```

## Contributing
Feel free to submit issues and enhancement requests.
