import pyaudio
import keyboard
import threading
import whisper
import platform
import subprocess
import time
import queue
import numpy as np
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

# Configure logging
log_file = Path(__file__).parent / 'logs.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceOver:
    def __init__(self):
        logger.info("Initializing VoiceOver")
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.audio_queue = queue.Queue()
        logger.info("Loading Whisper model (tiny)")
        self.whisper_model = whisper.load_model("tiny")
        self.system = platform.system()
        logger.info(f"Running on {self.system}")
        
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

    def start_recording(self, input_device):
        """Start recording audio from the selected input device."""
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000  # Whisper expects 16kHz

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CHUNK
        )
        
        logger.info(f"Started recording from device {input_device}")
        self.recording = True
        
        while self.recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_data)
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                break

        self.stream.stop_stream()
        self.stream.close()
        logger.info("Stopped recording")

    def process_audio(self):
        """Process audio chunks and transcribe using Whisper."""
        audio_buffer = []
        silence_threshold = 0.01
        min_audio_length = 0.5  # minimum seconds of audio to process
        samples_per_sec = 16000
        
        while self.recording or not self.audio_queue.empty():
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_buffer.extend(chunk)

            # Process audio if we have enough samples
            if len(audio_buffer) > samples_per_sec * min_audio_length:
                audio_array = np.array(audio_buffer)
                
                # Only process if audio is not silence
                if np.abs(audio_array).mean() > silence_threshold:
                    try:
                        result = self.whisper_model.transcribe(
                            audio_array, 
                            language='en',
                            fp16=False
                        )
                        if result["text"].strip():
                            print(f"Transcription: {result['text'].strip()}")
                            logger.info(f"Transcribed: {result['text'].strip()}")
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {e}")
                
                # Clear the buffer
                audio_buffer = []

        logger.info("Stopped processing audio")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'audio'):
            logger.info("Terminating PyAudio")
            self.audio.terminate()

if __name__ == "__main__":
    logger.info("Starting VoiceOver application")
    
    try:
        voiceover = VoiceOver()
        input_device, output_device = voiceover.select_devices()
        
        if input_device is not None and output_device is not None:
            logger.info(f"Successfully initialized with input device {input_device} and output device {output_device}")
            print("\nPress and hold 'V' to start recording. Release to stop. Press Ctrl+C to exit.")
            
            # Create threads for recording and processing
            record_thread = None
            process_thread = None
            
            def on_press(event):
                if event.name == 'v' and not voiceover.recording:
                    logger.info("'V' key pressed - starting recording")
                    voiceover.recording = True
                    
                    # Start recording thread
                    nonlocal record_thread, process_thread
                    record_thread = threading.Thread(target=voiceover.start_recording, args=(input_device,))
                    process_thread = threading.Thread(target=voiceover.process_audio)
                    
                    record_thread.start()
                    process_thread.start()

            def on_release(event):
                if event.name == 'v' and voiceover.recording:
                    logger.info("'V' key released - stopping recording")
                    voiceover.recording = False
                    
                    # Wait for threads to complete
                    if record_thread and process_thread:
                        record_thread.join()
                        process_thread.join()

            # Register key event handlers
            keyboard.on_press(on_press)
            keyboard.on_release(on_release)

            # Keep the main thread alive
            keyboard.wait('esc')
            
        else:
            logger.error("Failed to select valid devices")
            print("Failed to select valid devices. Exiting.")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred. Check logs.txt for details.")
    finally:
        # Ensure recording is stopped
        if hasattr(voiceover, 'recording'):
            voiceover.recording = False
