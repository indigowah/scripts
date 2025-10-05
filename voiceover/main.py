#!/usr/bin/env python3
import pyaudio
import numpy as np
import whisper
import keyboard
import yaml
import argparse
import threading
import queue
import time
import sounddevice as sd
import espeak
from scipy.io import wavfile
import os
from typing import Dict, Any

class VoiceChanger:
    def __init__(self):
        self.config_file = "config.yaml"
        self.pa = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.running = False
        self.silence_threshold = 0.02
        self.silence_count = 0
        self.silence_frames = 30  # frames of silence to determine end of sentence
        self.load_or_create_config()

    def load_or_create_config(self) -> None:
        """Load existing config or create new one with user input."""
        if os.path.exists(self.config_file) and not self.should_reconfigure():
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.configure_settings()
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f)

    def should_reconfigure(self) -> bool:
        """Check if the script was launched with -r flag."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--reconfigure', action='store_true',
                          help='Reconfigure settings')
        args = parser.parse_args()
        return args.reconfigure

    def list_audio_devices(self) -> None:
        """List all available audio devices."""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            print(f"Index {i}: {dev_info['name']}")
            print(f"    Input channels: {dev_info['maxInputChannels']}")
            print(f"    Output channels: {dev_info['maxOutputChannels']}")
            print("-" * 50)

    def test_espeak_settings(self, settings: Dict[str, Any]) -> None:
        """Test espeak settings with a sample text."""
        espeak.init()
        for k, v in settings.items():
            if hasattr(espeak, f"set_{k}"):
                getattr(espeak, f"set_{k}")(v)
        
        test_text = "This is a test of the current espeak settings."
        print("\nPlaying test voice...")
        espeak.synth(test_text)
        time.sleep(2)  # Wait for speech to complete

    def configure_settings(self) -> Dict[str, Any]:
        """Configure all settings with user input."""
        config = {}
        
        # Audio device selection
        self.list_audio_devices()
        
        while True:
            try:
                input_device = int(input("\nSelect input device index: "))
                if 0 <= input_device < self.pa.get_device_count():
                    config['input_device'] = input_device
                    break
                print("Invalid device index!")
            except ValueError:
                print("Please enter a valid number!")

        while True:
            try:
                output_device = int(input("Select output device index: "))
                if 0 <= output_device < self.pa.get_device_count():
                    config['output_device'] = output_device
                    break
                print("Invalid device index!")
            except ValueError:
                print("Please enter a valid number!")

        # Whisper model selection
        print("\nAvailable Whisper models:")
        models = ['tiny', 'base', 'small', 'medium', 'large']
        for i, model in enumerate(models):
            print(f"{i}: {model}")
        
        while True:
            try:
                model_idx = int(input("\nSelect Whisper model index: "))
                if 0 <= model_idx < len(models):
                    config['whisper_model'] = models[model_idx]
                    break
                print("Invalid model index!")
            except ValueError:
                print("Please enter a valid number!")

        # eSpeak settings
        print("\nConfigure eSpeak settings:")
        config['espeak'] = {
            'rate': int(input("Speech rate (words per minute, default 175): ") or 175),
            'pitch': int(input("Pitch (0-100, default 50): ") or 50),
            'volume': int(input("Volume (0-100, default 100): ") or 100)
        }

        # Test eSpeak settings
        while True:
            self.test_espeak_settings(config['espeak'])
            if input("\nAre these settings okay? (y/n): ").lower() == 'y':
                break
            print("\nLet's adjust the settings:")
            config['espeak']['rate'] = int(input("Speech rate: ") or config['espeak']['rate'])
            config['espeak']['pitch'] = int(input("Pitch: ") or config['espeak']['pitch'])
            config['espeak']['volume'] = int(input("Volume: ") or config['espeak']['volume'])

        return config

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio input stream error: {status}")
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        energy = np.mean(np.abs(audio_data))
        
        if energy > self.silence_threshold:
            self.silence_count = 0
            self.audio_queue.put(audio_data)
        else:
            self.silence_count += 1
            if self.silence_count >= self.silence_frames:
                self.audio_queue.put(None)  # Signal end of sentence
                self.silence_count = 0
        
        return (in_data, pyaudio.paContinue)

    def transcribe_audio(self):
        """Thread function for transcribing audio using Whisper."""
        model = whisper.load_model(self.config['whisper_model'])
        accumulated_audio = []

        while self.running:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:  # End of sentence
                if accumulated_audio:
                    audio_data = np.concatenate(accumulated_audio)
                    result = model.transcribe(audio_data)
                    self.text_queue.put(result['text'])
                    accumulated_audio = []
            else:
                accumulated_audio.append(audio_chunk)

    def text_to_speech(self):
        """Thread function for converting text to speech using eSpeak."""
        espeak.init()
        for k, v in self.config['espeak'].items():
            if hasattr(espeak, f"set_{k}"):
                getattr(espeak, f"set_{k}")(v)

        while self.running:
            text = self.text_queue.get()
            if text:
                espeak.synth(text)

    def run(self):
        """Main function to run the voice changer."""
        self.running = True

        # Set up audio stream
        stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            output=False,
            input_device_index=self.config['input_device'],
            stream_callback=self.audio_callback,
            frames_per_buffer=1024
        )

        # Start threads
        transcribe_thread = threading.Thread(target=self.transcribe_audio)
        speech_thread = threading.Thread(target=self.text_to_speech)
        
        transcribe_thread.start()
        speech_thread.start()

        print("\nVoice changer is running! Press 'end' to quit.")
        stream.start_stream()

        while self.running:
            if keyboard.is_pressed('end'):
                self.running = False
                break
            time.sleep(0.1)

        # Cleanup
        stream.stop_stream()
        stream.close()
        self.pa.terminate()
        transcribe_thread.join()
        speech_thread.join()

if __name__ == "__main__":
    voice_changer = VoiceChanger()
    voice_changer.run()
