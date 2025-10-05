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
import sys
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

    def verify_tts_available(self, command):
        """Verify if TTS command is available."""
        try:
            if self.system == "Windows":
                # Check common installation paths for espeak on Windows
                espeak_paths = [
                    "C:\\Program Files\\eSpeak\\command_line\\espeak.exe",
                    "C:\\Program Files (x86)\\eSpeak\\command_line\\espeak.exe",
                    "espeak"  # Try system PATH
                ]
                for path in espeak_paths:
                    try:
                        subprocess.run([path, "--version"], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                        return path
                    except FileNotFoundError:
                        continue
                raise FileNotFoundError("espeak not found in common locations")
            else:
                # For macOS and Linux, check if command exists
                subprocess.run([command, "--version"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
                return command
        except Exception as e:
            raise FileNotFoundError(f"TTS command '{command}' not available: {e}")

    def initialize_tts(self):
        """Initialize the appropriate TTS engine based on the platform."""
        if self.system == "Darwin":  # macOS
            try:
                self.tts_command = self.verify_tts_available("say")
                logger.info("Initialized macOS 'say' command for TTS")
            except FileNotFoundError:
                logger.error("macOS 'say' command not available")
                raise
        else:  # Windows/Linux
            try:
                self.tts_command = self.verify_tts_available("espeak")
                logger.info("Initialized espeak for TTS")
            except FileNotFoundError:
                logger.error("espeak not found. Please install espeak:")
                if self.system == "Windows":
                    logger.error("Download espeak from: http://espeak.sourceforge.net/")
                    logger.error("Or install with: winget install espeak")
                else:
                    logger.error("Install with: sudo apt-get install espeak")
                raise

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
        CHUNK = 512  # Reduced chunk size for lower latency
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000  # Whisper expects 16kHz

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback  # Use callback for non-blocking
        )
        
        logger.info(f"Started recording from device {input_device}")
        self.recording = True
        self.stream.start_stream()
        
        while self.recording:
            time.sleep(0.001)  # Tiny sleep to prevent CPU overload
            
        self.stream.stop_stream()
        self.stream.close()
        logger.info("Stopped recording")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio processing."""
        if self.recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        """Process audio chunks and transcribe using Whisper."""
        audio_buffer = []
        silence_threshold = 0.01
        min_audio_length = 0.5  # Increased to capture more complete phrases
        max_audio_length = 1.0  # Increased to allow for complete sentences
        samples_per_sec = 16000
        last_process_time = time.time()
        silence_duration = 0.3  # Duration of silence to consider end of phrase
        last_voice_time = time.time()
        self.tts_queue = queue.Queue()
        self.tts_busy = False
        
        # Start TTS worker thread
        tts_worker = threading.Thread(target=self._tts_worker)
        tts_worker.daemon = True
        tts_worker.start()
        
        while self.recording or not self.audio_queue.empty():
            current_time = time.time()
            
            # Process any available chunks
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                audio_buffer.extend(chunk)

            buffer_duration = len(audio_buffer) / samples_per_sec
            
            # Check if we have voice in the current buffer
            if audio_buffer:
                current_amplitude = np.abs(np.array(audio_buffer)).mean()
                if current_amplitude > silence_threshold:
                    last_voice_time = current_time
            
            # Process audio if we have enough samples and either:
            # 1. Max length reached, or
            # 2. Silence detected for long enough
            should_process = (
                buffer_duration >= min_audio_length and
                (buffer_duration >= max_audio_length or
                 current_time - last_voice_time >= silence_duration)
            )
            
            if should_process and audio_buffer:
                audio_array = np.array(audio_buffer)
                
                # Only process if audio is not silence
                if np.abs(audio_array).mean() > silence_threshold:
                    try:
                        # Optimize Whisper transcription for speed
                        result = self.whisper_model.transcribe(
                            audio_array,
                            language='en',
                            fp16=False,
                            without_timestamps=True,
                            condition_on_previous_text=True,  # Enable for better sentence completion
                            temperature=0.0
                        )
                        
                        transcribed_text = result["text"].strip()
                        if transcribed_text:
                            print(f"Transcription: {transcribed_text}")
                            logger.info(f"Transcribed: {transcribed_text}")
                            self.tts_queue.put(transcribed_text)
                            
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {e}")
                
                # Clear the buffer and update time
                audio_buffer = []
                last_process_time = current_time
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)

    def _tts_worker(self):
        """Worker thread to process TTS queue."""
        while self.recording or not self.tts_queue.empty():
            try:
                text = self.tts_queue.get(timeout=0.5)
                self._run_tts(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}")
                
    def _run_tts(self, text):
        """Run TTS with proper device routing."""
        try:
            if self.system == "Darwin":
                cmd = [self.tts_command, text]
            else:
                cmd = [
                    self.tts_command,
                    "-s", "175",  # Speed
                    "-v", "en",   # Voice
                    "-a", "150",  # Amplitude
                    "--stdout"    # Output to stdout instead of default audio
                ]
                if hasattr(self, 'output_device'):
                    # Add text last for espeak
                    cmd.extend([text])
                
            if self.system == "Windows":
                # Use PowerShell to redirect audio to specific device
                full_cmd = [
                    "powershell.exe",
                    "-Command",
                    f"$audio = New-Object System.Media.SoundPlayer; " +
                    f"$audio.PlayDevice = {self.output_device}; " +
                    f"& {' '.join(cmd)}"
                ]
                cmd = full_cmd
            
            subprocess.run(cmd, check=True)
            logger.info(f"Sent to TTS: {text}")
            thread_manager.increment_transcription_count()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"TTS command failed: {e}")
        except Exception as e:
            logger.error(f"Error with TTS: {e}")
            
        # Add a small delay between TTS outputs to prevent overlap
        time.sleep(0.1)

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
        self.running = True
        self.session_start_time = time.time()
        self.transcription_count = 0

    def start_threads(self):
        """Start recording and processing threads."""
        logger.info("Starting continuous recording and processing")
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
        """Stop recording and processing threads."""
        if self.voiceover.recording:
            logger.info("Stopping recording and processing")
            self.voiceover.recording = False
            
            if self.record_thread and self.process_thread:
                self.record_thread.join()
                self.process_thread.join()

    def increment_transcription_count(self):
        self.transcription_count += 1

    def save_session_summary(self):
        """Save session summary to log file."""
        session_duration = time.time() - self.session_start_time
        hours, remainder = divmod(int(session_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        summary = f"""
{'-' * 50}
Session Summary
{'-' * 50}
Duration: {hours:02d}:{minutes:02d}:{seconds:02d}
Total Transcriptions: {self.transcription_count}
Whisper Model Used: {self.voiceover.whisper_model.model_type}
System Platform: {self.voiceover.system}
TTS Engine: {self.voiceover.tts_command}
{'-' * 50}
"""
        logger.info(summary)
        
        # Save summary to a separate file
        summary_file = Path(__file__).parent / f'session_summary_{time.strftime("%Y%m%d_%H%M%S")}.txt'
        try:
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Session summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving session summary: {e}")

    def shutdown(self):
        """Gracefully shutdown the audio manager."""
        logger.info("Initiating shutdown...")
        self.running = False
        self.voiceover.recording = False
        self.save_session_summary()

if __name__ == "__main__":
    logger.info("Starting VoiceOver application")
    
    try:
        voiceover = VoiceOver()
    except FileNotFoundError as e:
        logger.error("Failed to initialize TTS engine")
        print("\nError: Text-to-Speech engine not found!")
        print("Please install the required TTS engine and try again.")
        sys.exit(1)
        
    try:
        input_device, output_device = voiceover.select_devices()
        
        if input_device is not None and output_device is not None:
            logger.info(f"Successfully initialized with input device {input_device} and output device {output_device}")
            print("\nPress and hold 'V' to start recording. Release to stop. Press 'Esc' to exit.")
            
            # Create thread manager
            thread_manager = AudioThreadManager(voiceover, input_device)
            
            # Start recording immediately
            thread_manager.start_threads()
            
            def on_end_press(_):
                print("\nEnding session and saving logs...")
                thread_manager.shutdown()
                return False  # Stop listener
            
            # Register end key handler
            keyboard.on_press_key('end', on_end_press)
            
            print("\nRecording started...")
            print("\nControls:")
            print("- Press 'End' to save and exit")
            print("- Press 'Esc' for emergency exit (logs may not save properly)\n")

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
