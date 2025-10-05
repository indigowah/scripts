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
import pyttsx3
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
        self.engine = pyttsx3.init()
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

    def test_tts_settings(self, settings: Dict[str, Any]) -> None:
        """Test text-to-speech settings with a sample text."""
        self.engine.setProperty('rate', settings['rate'])
        self.engine.setProperty('volume', settings['volume'] / 100.0)  # Convert to 0-1 range
        
        voices = self.engine.getProperty('voices')
        if settings.get('voice_idx', 0) < len(voices):
            self.engine.setProperty('voice', voices[settings['voice_idx']].id)
        
        test_text = "This is a test of the current voice settings."
        print("\nPlaying test voice...")
        self.engine.say(test_text)
        self.engine.runAndWait()

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

        # TTS settings
        print("\nConfigure Text-to-Speech settings:")
        config['tts'] = {
            'rate': int(input("Speech rate (words per minute, default 175): ") or 175),
            'volume': int(input("Volume (0-100, default 100): ") or 100)
        }

        # Voice selection
        voices = self.engine.getProperty('voices')
        print("\nAvailable voices:")
        for idx, voice in enumerate(voices):
            print(f"{idx}: {voice.name}")
        
        while True:
            try:
                voice_idx = int(input("\nSelect voice index: "))
                if 0 <= voice_idx < len(voices):
                    config['tts']['voice_idx'] = voice_idx
                    break
                print("Invalid voice index!")
            except ValueError:
                print("Please enter a valid number!")

        # Test TTS settings
        while True:
            self.test_tts_settings(config['tts'])
            if input("\nAre these settings okay? (y/n): ").lower() == 'y':
                break
            print("\nLet's adjust the settings:")
            config['tts']['rate'] = int(input("Speech rate: ") or config['tts']['rate'])
            config['tts']['volume'] = int(input("Volume: ") or config['tts']['volume'])
            voice_idx = int(input("Voice index: ") or config['tts']['voice_idx'])
            if 0 <= voice_idx < len(voices):
                config['tts']['voice_idx'] = voice_idx

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
                    transcribed_text = result['text'].strip()
                    if transcribed_text:  # Only process non-empty text
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"\n[{timestamp}] Heard: {transcribed_text}")
                        self.text_queue.put(transcribed_text)
                    accumulated_audio = []
            else:
                accumulated_audio.append(audio_chunk)

    def text_to_speech(self):
        """Thread function for converting text to speech using pyttsx3."""
        # Configure TTS settings
        self.engine.setProperty('rate', self.config['tts']['rate'])
        self.engine.setProperty('volume', self.config['tts']['volume'] / 100.0)
        
        voices = self.engine.getProperty('voices')
        voice_idx = self.config['tts'].get('voice_idx', 0)
        if voice_idx < len(voices):
            self.engine.setProperty('voice', voices[voice_idx].id)

        while self.running:
            text = self.text_queue.get()
            if text:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()

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
