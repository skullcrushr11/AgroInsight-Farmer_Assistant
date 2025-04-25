import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import os
import time
import torch
import subprocess
import sys

def check_ffmpeg():
    """Verify ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("ffmpeg is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Please install it:")
        print("- Run 'choco install ffmpeg' (with Chocolatey)")
        print("- Or download from https://ffmpeg.org and add to PATH")
        print("Example manual install:")
        print("1. Download ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to System PATH")
        sys.exit(1)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished.")
    return audio, sample_rate

def save_audio(audio, sample_rate):
    """Save audio to a WAV file in the current directory."""
    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    audio_file = os.path.join(output_dir, "recorded_audio.wav")
    
    try:
        wavfile.write(audio_file, sample_rate, audio)
        print(f"Audio saved to: {audio_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Failed to save audio file: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        raise

def transcribe_audio(audio_file, model_name="medium", language="en"):
    """Transcribe audio using Whisper with specified language."""
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        start_time = time.time()
        model = whisper.load_model(model_name).to(device)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        start_time = time.time()
        result = model.transcribe(audio_file, language=language)  
        transcribe_time = time.time() - start_time
        print(f"Transcription took {transcribe_time:.2f} seconds")
        
        return {
            "text": result["text"],
            "language": language,
            "model_name": model_name
        }
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA out of memory! Try 'base' model or clear GPU memory.")
        else:
            print(f"Error loading/transcribing: {str(e)}")
        return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    
    check_ffmpeg()
    
    
    duration = 5  
    model_name = "medium"  
    language = "hi"  
    
    
    try:
        audio, sample_rate = record_audio(duration)
    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        return
    
    
    try:
        audio_file = save_audio(audio, sample_rate)
    except Exception as e:
        print(f"Failed to save audio: {str(e)}")
        return
    
    
    try:
        result = transcribe_audio(audio_file, model_name, language)
        if result:
            print(f"\nResults:")
            print(f"Specified language: {result['language']}")
            print(f"Transcribed text: {result['text']}")
            
        else:
            print("Transcription failed. Trying 'base' model...")
            result = transcribe_audio(audio_file, "base", language)
            if result:
                print(f"\nFallback Results:")
                print(f"Specified language: {result['language']}")
                print(f"Transcribed text: {result['text']}")
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
    
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print("Audio file deleted.")

if __name__ == "__main__":
    main()