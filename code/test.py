import sys
import time
import numpy as np
import logging
from RealtimeSTT import AudioToTextRecorder

# Setup basic logging
logging.basicConfig(level=logging.WARNING)

def calculate_volume(chunk):
    """Calculates the volume (RMS) of an audio chunk."""
    # Convert bytes to numpy array
    audio_data = np.frombuffer(chunk, dtype=np.int16)
    # Calculate RMS (Root Mean Square)
    rms = np.sqrt(np.mean(audio_data**2))
    return rms

def on_recorded_chunk(chunk):
    """Callback: Visualizes audio volume to verify mic input."""
    vol = calculate_volume(chunk)
    # Create a simple visual bar
    bar_length = int(vol / 50)  # Adjust divisor based on your mic sensitivity
    bar = "|" * bar_length
    # Print overwriting the same line to create an animation effect
    sys.stdout.write(f"\r\033[90mVolume: [{bar:<20}] {int(vol)}\033[0m")
    sys.stdout.flush()

def on_vad_start():
    print("\n\033[92m[VAD] Voice Activity Detected! (Recording Started)\033[0m")

def on_vad_stop():
    print("\n\033[91m[VAD] Silence Detected. (Recording Stopped)\033[0m")

def on_realtime_update(text):
    print(f"\r\033[96m[TRANS] Partial: {text}\033[0m")

def on_final_result(text):
    print(f"\r\033[93m[FINAL] Result: {text}\033[0m\n")

print("Initializing RealtimeSTT... (This might take a moment to load models)")

try:
    # We set debug_mode=True to see internal errors if any
    # We enable realtime transcription
    recorder = AudioToTextRecorder(
        spinner=False,
        model="tiny.en", # Use tiny model for fastest initialization test
        language="en",
        enable_realtime_transcription=True,
        on_recorded_chunk=on_recorded_chunk, # To see volume
        on_vad_start=on_vad_start,           # To see VAD trigger
        on_vad_stop=on_vad_stop,
        on_realtime_transcription_update=on_realtime_update,
        on_transcription_start=None, # Clean output
        debug_mode=False, # Set to True if you want to see massive internal logs
        # Adjust sensitivity if VAD isn't triggering
        silero_sensitivity=0.05, # Very sensitive (0.01 - 1.0)
        webrtc_sensitivity=3,    # 3 is most aggressive suppression, 0 is least.
        input_device_index=2, # Set this index if you have multiple mics!
    )

    print("\n✅ Recorder Initialized!")
    print("🎤 Please speak into your microphone.")
    print("📊 If the volume bar moves, your mic is working.")
    print("⚡ If '[VAD]' appears, speech detection is working.")
    print("📝 If '[TRANS]' appears, the model is working.")
    print("-------------------------------------------------------")

    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting...")
    if 'recorder' in locals():
        recorder.shutdown()
except Exception as e:
    print(f"\n❌ Error: {e}")