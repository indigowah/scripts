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
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self):
        logger.info("Initializing VoiceOver")
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.system = platform.system()
        logger.info(f"Running on {self.system}")
        
        # Select Whisper model
        self.whisper_model = self.select_whisper_model()
        
        # Initialize TTS engine
        self.initialize_tts()

    def initialize_tts(self):
        """Initialize the appropriate TTS engine based on the platform."""
        if self.system == "Darwin":  # macOS
            self.tts_command = "say"
            logger.info("Initialized macOS 'say' command for TTS")
        else:  # Windows/Linux
            self.tts_command = "espeak"
            logger.info("Initialized espeak for TTS")

    def select_whisper_model(self) -> whisper.Whisper:
        """Let user select the Whisper model to use."""
        print("\nAvailable Whisper Models:")
        print("-" * 50)
        for i, model in enumerate(self.AVAILABLE_MODELS):
            print(f"{i + 1}. {model}")
        print("-" * 50)
        
        while True:
            try:
                choice = int(input("\nEnter the number of the model you want to use (1-5): "))
                if 1 <= choice <= len(self.AVAILABLE_MODELS):
                    model_name = self.AVAILABLE_MODELS[choice - 1]
                    logger.info(f"Loading Whisper model ({model_name})")
                    return whisper.load_model(model_name)
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
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
                        transcribed_text = result["text"].strip()
                        if transcribed_text:
                            print(f"Transcription: {transcribed_text}")
                            logger.info(f"Transcribed: {transcribed_text}")
                            
                            # Send to TTS
                            try:
                                if self.system == "Darwin":
                                    subprocess.run([self.tts_command, transcribed_text])
                                else:
                                    subprocess.run([self.tts_command, transcribed_text])
                                logger.info(f"Sent to TTS: {transcribed_text}")
                            except Exception as e:
                                logger.error(f"Error with TTS: {e}")
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

class AudioThreadManager:
    def __init__(self, voiceover, input_device):
        self.voiceover = voiceover
        self.input_device = input_device
        self.record_thread = None
        self.process_thread = None

    def start_threads(self):
        if not self.voiceover.recording:
            logger.info("'V' key pressed - starting recording")
            self.voiceover.recording = True
            
            self.record_thread = threading.Thread(
                target=self.voiceover.start_recording, 
                args=(self.input_device,)
            )
            self.process_thread = threading.Thread(
                target=self.voiceover.process_audio
            )
            
            self.record_thread.start()
            self.process_thread.start()

    def stop_threads(self):
        if self.voiceover.recording:
            logger.info("'V' key released - stopping recording")
            self.voiceover.recording = False
            
            if self.record_thread and self.process_thread:
                self.record_thread.join()
                self.process_thread.join()

if __name__ == "__main__":
    logger.info("Starting VoiceOver application")
    
    try:
        voiceover = VoiceOver()
        input_device, output_device = voiceover.select_devices()
        
        if input_device is not None and output_device is not None:
            logger.info(f"Successfully initialized with input device {input_device} and output device {output_device}")
            print("\nPress and hold 'V' to start recording. Release to stop. Press 'Esc' to exit.")
            
            # Create thread manager
            thread_manager = AudioThreadManager(voiceover, input_device)
            
            # Set up key handlers
            keyboard.on_press_key('v', lambda _: thread_manager.start_threads())
            keyboard.on_release_key('v', lambda _: thread_manager.stop_threads())

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
