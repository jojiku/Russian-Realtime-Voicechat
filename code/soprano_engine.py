"""
Soprano TTS Engine for RealtimeTTS

Requires:
- pip install soprano-tts torch scipy
"""
from soprano import SopranoTTS
from base_engine import BaseEngine
from queue import Queue
from typing import Union
import numpy as np
import traceback
import pyaudio
import torch
import time


class SopranoVoice:
    """Wrapper for Soprano voice configuration."""
    
    def __init__(self, name: str = "default"):
        self.name = name
    
    def __repr__(self):
        return f"SopranoVoice: {self.name}"


class SopranoEngine(BaseEngine):
    """
    A text-to-speech (TTS) engine utilizing the Soprano model.
    """

    def __init__(
        self,
        voice: Union[str, SopranoVoice] = "default",
        chunk_size: int = 15,  # Keep >10 to prevent artifacts
        backend: str = 'auto',
        device: str = 'cuda',
        cache_size_mb: int = 10, 
        decoder_batch_size: int = 1,
        output_sample_rate: int = None,
        model_sample_rate: int = None,
        playback_chunk_size: int = 4096, 
        debug: bool = False,
        apply_fades: bool = True,
        fade_duration_ms: int = 10
    ):
        super().__init__()
        self.engine_name = "soprano"
        self.queue = Queue()
        self.debug = debug
        self.apply_fades = apply_fades
        self.fade_duration_ms = fade_duration_ms
        
        self.chunk_size = chunk_size
        self.backend = backend
        self.device = device
        self.output_sample_rate = output_sample_rate 
        
        # Auto-detect device sample rate if not specified
        if self.output_sample_rate is None:
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                device_info = p.get_default_output_device_info()
                self.output_sample_rate = int(device_info['defaultSampleRate'])
                p.terminate()
            except:
                self.output_sample_rate = 44100
        
        if self.debug:
            print(f"[SopranoEngine] Loading model with backend='{backend}'...")

        self.model = SopranoTTS(
            backend=backend, 
            device=device,
            cache_size_mb=cache_size_mb,
            decoder_batch_size=decoder_batch_size
        )

        # --- DYNAMIC SAMPLE RATE DETECTION ---
        # Default fallback for Soprano is usually 32000Hz
        self.soprano_sample_rate = 32000 
        
        if model_sample_rate:
            self.soprano_sample_rate = model_sample_rate
        elif hasattr(self.model, "sample_rate"):
            self.soprano_sample_rate = self.model.sample_rate
        elif hasattr(self.model, "config") and hasattr(self.model.config, "sampling_rate"):
            self.soprano_sample_rate = self.model.config.sampling_rate
        # REMOVED the crashing check for self.model.decoder.config
            
        self.current_voice = voice if isinstance(voice, str) else voice.name
        
        if self.debug:
            print(f"[SopranoEngine] Ready.")
            print(f"  - Source Rate (Model): {self.soprano_sample_rate} Hz")
            print(f"  - Target Rate (Device): {self.output_sample_rate} Hz")


    def get_stream_info(self):
        return (pyaudio.paFloat32, 1, self.output_sample_rate)

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, orig_sr)
            up = target_sr // g
            down = orig_sr // g
            return resample_poly(audio, up, down)
        except ImportError:
            if self.debug:
                print("[SopranoEngine] Error: scipy not installed. Audio will be wrong pitch.")
            return audio

    def synthesize(self, text: str) -> bool:
        """
        True Streaming Synthesis with On-The-Fly Resampling
        """
        # Ensure the engine is allowed to run
        self.stop_synthesis_event.clear()
        start_total = time.time()
        try:
            if self.debug:
                print(f"\n[SopranoEngine] Streaming: '{text}'")

            # Start the stream iterator from Soprano
            stream = self.model.infer_stream(text, chunk_size=self.chunk_size)
            
            first_chunk = True
            # Iterate through chunks as they arrive (True Streaming)
            for chunk in stream:
                # Check for stop signal
                if self.stop_synthesis_event.is_set():
                    if self.debug: print("[SopranoEngine] Stop event.")
                    return False

                # 1. Convert Tensor to Numpy
                audio_chunk = chunk.cpu().numpy()
                
                # 2. Resample immediately if rates differ (32000 -> 48000/24000)
                if self.output_sample_rate != self.soprano_sample_rate:
                    audio_chunk = self._resample_audio(
                        audio_chunk, 
                        self.soprano_sample_rate, 
                        self.output_sample_rate
                    )

                # 3. Apply Fade-In (Only on the very first chunk to avoid clicks at start)
                if first_chunk and self.apply_fades:
                    fade_samples = int(self.output_sample_rate * self.fade_duration_ms / 1000)
                    if len(audio_chunk) > fade_samples:
                        fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)
                        audio_chunk[:fade_samples] *= fade_in
                        latency = time.time() - start_total
                        print(f"[Internal Debug] Time to first audio chunk: {latency * 1000:.2f} ms")
                    first_chunk = False
                
                # 4. Push to Queue IMMEDIATELY
                # We do not accumulate. We push bytes to the queue so RealtimeTTS plays them NOW.
                self.queue.put(audio_chunk.tobytes())
                
                # Track duration logic for BaseEngine
                self.audio_duration += len(audio_chunk) / self.output_sample_rate

            return True
            
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            print(f"[SopranoEngine] Error: {e}")
            return False

    def set_voice(self, voice: Union[str, SopranoVoice]):
        self.current_voice = voice.name if isinstance(voice, SopranoVoice) else voice

    def set_voice_parameters(self, **voice_parameters):
        if "chunk_size" in voice_parameters:
            self.chunk_size = voice_parameters["chunk_size"]
        if "output_sample_rate" in voice_parameters:
            self.output_sample_rate = voice_parameters["output_sample_rate"]

    def get_voices(self):
        return [SopranoVoice("default")]

    def shutdown(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__=="__main__":
    from RealtimeTTS import TextToAudioStream
    
    # Test Config
    engine = SopranoEngine(
        debug=True, 
        chunk_size=15, 
        output_sample_rate=24000 # Simulating your server setup
    )
    
    stream = TextToAudioStream(
        engine,
        frames_per_buffer=1024,
        playout_chunk_size=1024
    )
    print("Starting stream...")
    stream.feed("This text should start playing almost immediately, without waiting for the end.").play()