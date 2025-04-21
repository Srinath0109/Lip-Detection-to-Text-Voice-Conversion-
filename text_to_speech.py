import subprocess

class TextToSpeechEngine:
    def speak(self, text):
        try:
            subprocess.run(['say', text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
