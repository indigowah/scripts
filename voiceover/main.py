import pyaudio
import keyboard
import threading
import whisper
import platform
import subprocess
import time
import queue
import numpy as np
from typing import Optional

class VoiceOver:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.whisper_model = whisper.load_model("base")
        self.system = platform.system()
        
    def list_audio_devices(self):
        """List all available audio input and output devices."""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            device_type = []
            
            if device_info['maxInputChannels'] > 0:
                device_type.append("INPUT")
            if device_info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
                
            print(f"Device ID: {i}")
            print(f"Name: {device_info['name']}")
            print(f"Type: {' & '.join(device_type)}")
            print(f"Default Sample Rate: {int(device_info['defaultSampleRate'])}Hz")
            print("-" * 50)

    def select_devices(self) -> tuple[int, int]:
        """Let user select input and output devices."""
        self.list_audio_devices()
        
        try:
            input_id = int(input("\nEnter the ID number for your INPUT device: "))
            output_id = int(input("Enter the ID number for your OUTPUT device: "))
            
            # Validate selections
            input_info = self.audio.get_device_info_by_index(input_id)
            output_info = self.audio.get_device_info_by_index(output_id)
            
            if input_info['maxInputChannels'] == 0:
                raise ValueError("Selected input device has no input channels")
            if output_info['maxOutputChannels'] == 0:
                raise ValueError("Selected output device has no output channels")
                
            return input_id, output_id
            
        except (ValueError, OSError) as e:
            print(f"Error selecting devices: {e}")
            return None, None

if __name__ == "__main__":
    voiceover = VoiceOver()
    input_device, output_device = voiceover.select_devices()
    
    if input_device is not None and output_device is not None:
        print(f"\nSelected input device ID: {input_device}")
        print(f"Selected output device ID: {output_device}")
    else:
        print("Failed to select valid devices. Exiting.")
