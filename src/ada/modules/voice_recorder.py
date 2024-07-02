"""
voice_recorder.py - Voice Recognition and Command Processing Module for ADA AI Assistant

This module implements a VoiceRecorder class that provides continuous voice recognition
and command processing capabilities for the ADA (Advanced Digital Assistant) AI Assistant.
It utilizes the Vosk speech recognition library and sounddevice for audio input handling.

Key Features:
1. Continuous Voice Recognition: Listens to audio input in real-time.
2. Keyword Activation: Starts recording when a specific activation keyword is detected.
3. Command Processing: Captures and processes voice commands between activation and end keywords.
4. Customizable Keywords: Allows setting custom activation, end, and stop keywords.
5. Transcription: Provides transcripts of recognized speech.

Main Components:
- VoiceRecorder: The main class that handles voice recognition and command processing.
  - __init__: Initializes the recorder with model path and keywords.
  - callback: Callback function for the audio stream to queue audio data.
  - continuous_listen: Main loop for continuous audio processing.
  - process_result: Processes each recognized speech segment.
  - start_interaction: Begins recording an interaction.
  - stop_interaction: Ends recording and processes the captured command.
  - process_command: Placeholder for command processing logic.

Usage:
    recorder = VoiceRecorder('./path/to/vosk/model')
    recorder.continuous_listen()

Dependencies:
- sounddevice: For capturing audio from the microphone.
- numpy: For handling audio data arrays.
- vosk: For speech recognition.
- queue: For queuing audio data between callbacks and processing.

Note:
This module is designed to be used as part of the ADA AI Assistant system. It provides
the voice interface for interacting with the assistant. The actual command processing
logic should be implemented in the process_command method or integrated with the main
ADA system.

The Vosk model should be downloaded separately and the path to the model directory
should be provided when initializing the VoiceRecorder.

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import queue

import sounddevice as sd
import vosk


class VoiceRecorder:
    def __init__(
        self,
        model_path="model",
        device=None,
        activation_keyword="Hello Ada",
        end_keyword="thanks",
        stop_keyword="stop recording",
    ):
        self.model = vosk.Model(model_path)
        self.device = device
        self.activation_keyword = activation_keyword.lower()
        self.end_keyword = end_keyword.lower()
        self.stop_keyword = stop_keyword.lower()
        self.interaction_transcript = ""
        self.recording = False
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def continuous_listen(self):
        with sd.RawInputStream(
            callback=self.callback,
            device=self.device,
            dtype="int16",
            channels=1,
            samplerate=16000,
        ) as stream:
            rec = vosk.KaldiRecognizer(self.model, stream.samplerate)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    continue_listening = self.process_result(eval(result)["text"])
                    if not continue_listening:
                        print("Shutting down the listening process.")
                        break

    def process_result(self, transcript):
        print(f"Detected: {transcript}")
        if self.activation_keyword in transcript and not self.recording:
            self.start_interaction()
        elif self.end_keyword in transcript and self.recording:
            self.stop_interaction()
        elif self.stop_keyword in transcript:
            return False
        if self.recording:
            self.interaction_transcript += " " + transcript
        return True

    def start_interaction(self):
        print("Starting interaction ...")
        self.recording = True

    def stop_interaction(self):
        print("Stopping interaction ...")
        self.process_command(self.interaction_transcript)
        self.interaction_transcript = ""
        self.recording = False

    def process_command(self, transcript):
        # Process the recorded audio or perform actions based on the last command
        print(f"Processing command: {transcript}")


# Example usage:
if __name__ == "__main__":
    # Ensure you have a Vosk model directory.
    recorder = VoiceRecorder("./audio_models/vosk-model-en-us-0.22-lgraph")
    recorder.continuous_listen()
