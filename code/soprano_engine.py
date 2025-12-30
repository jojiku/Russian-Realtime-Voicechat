"""
Soprano TTS Engine for RealtimeTTS

Requires:
- pip install soprano-tts torch scipy
"""

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
    
    Soprano delivers high-fidelity 32 kHz audio with real-time streaming
    capabilities and ultra-low latency.
    
    This version pre-resamples to 48kHz to prevent artifacts during playback.
    """

    def __init__(
        self,
        voice: Union[str, SopranoVoice] = "default",
        chunk_size: int = 1,
        backend: str = 'auto',
        device: str = 'cuda',
        cache_size_mb: int = 10, 
        decoder_batch_size: int = 1,
        output_sample_rate: int = None,
        debug: bool = False,
        apply_fades: bool = True,
        fade_duration_ms: int = 10
    ):
        """
        Initializes the SopranoEngine.

        Args:
            voice (Union[str, SopranoVoice]): Voice identifier (currently only default).
            chunk_size (int): Number of tokens to generate per streaming chunk.
            backend (str): Backend for Soprano ('auto', 'cuda', 'cpu').
            device (str): Device to use for inference.
            cache_size_mb (int): Cache size in MB.
            decoder_batch_size (int): Batch size for decoder.
            output_sample_rate (int): Target sample rate for output. 
                If None, automatically detects device sample rate.
                Common values: 44100 (CD quality), 48000 (professional).
            debug (bool): Enable detailed debugging output.
            apply_fades (bool): Apply fade-in/out to prevent clicks between sentences.
            fade_duration_ms (int): Duration of fades in milliseconds.
        """
        super().__init__()
        self.engine_name = "soprano"
        self.queue = Queue()
        self.debug = debug
        self.apply_fades = apply_fades
        self.fade_duration_ms = fade_duration_ms
        
        self.chunk_size = chunk_size
        self.backend = backend
        self.device = device
        self.cache_size_mb = cache_size_mb
        self.decoder_batch_size = decoder_batch_size
        self.output_sample_rate = output_sample_rate  # Can be None for auto-detect
        self.soprano_sample_rate = 32000
        self._detected_sample_rate = None
        
        if self.debug:
            print(f"[SopranoEngine] Initializing with:")
            print(f"  - backend: {backend}")
            print(f"  - device: {device}")
            if output_sample_rate:
                print(f"  - output_sample_rate: {output_sample_rate} (manual)")
            else:
                print(f"  - output_sample_rate: auto-detect")
        
        from soprano import SopranoTTS
        self.model = SopranoTTS(
            backend=backend, 
            device=device,
            cache_size_mb=cache_size_mb,
            decoder_batch_size=decoder_batch_size
        )
        self.current_voice = voice if isinstance(voice, str) else voice.name
        
        if self.debug:
            print(f"[SopranoEngine] Model loaded successfully")
        

    def get_stream_info(self):
        """
        Provides the PyAudio stream configuration.

        Returns:
            tuple: (pyaudio.paFloat32, 1, sample_rate)
                   If output_sample_rate was set, returns that.
                   Otherwise returns 32000 and lets RealtimeTTS detect device rate.
        """
        if self.output_sample_rate is not None:
            return (pyaudio.paFloat32, 1, self.output_sample_rate)
        else:
            # Return native rate - device detection will happen in AudioStream
            return (pyaudio.paFloat32, 1, self.soprano_sample_rate)

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resamples audio using scipy's resample_poly for high quality.
        
        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        from scipy.signal import resample_poly
        
        # Calculate upsampling and downsampling factors
        from math import gcd
        g = gcd(target_sr, orig_sr)
        up = target_sr // g
        down = orig_sr // g
        
        if self.debug:
            print(f"[SopranoEngine] Resampling {orig_sr}→{target_sr} (up={up}, down={down})")
        
        return resample_poly(audio, up, down)

    def synthesize(self, text: str) -> bool:
        """
        Converts input text into speech audio chunks.
        
        Pre-resamples the complete audio to match device sample rate
        to avoid chunk boundary artifacts.

        Args:
            text (str): The text string to synthesize.

        Returns:
            bool: True if synthesis is successful, False otherwise.
        """
        start_time = time.time()
        
        try:
            if self.debug:
                print(f"\n[SopranoEngine] Synthesizing: '{text}'")
            
            # Generate streaming chunks
            chunks = self.model.infer_stream(text, chunk_size=self.chunk_size)
            
            # Buffer all chunks
            chunk_list = []
            for chunk in chunks:
                if self.stop_synthesis_event.is_set():
                    if self.debug:
                        print(f"[SopranoEngine] Stop event detected")
                    return False
                chunk_list.append(chunk)
            
            if not chunk_list:
                if self.debug:
                    print(f"[SopranoEngine] No audio generated")
                return False
            
            # Concatenate all chunks at native 32kHz
            full_audio = torch.cat(chunk_list)
            audio_float32 = full_audio.cpu().numpy()
            
            if self.debug:
                orig_duration = len(audio_float32) / self.soprano_sample_rate
                print(f"[SopranoEngine] Generated: {len(audio_float32)} samples @ 32kHz, {orig_duration:.2f}s")
            
            # Apply fades at native sample rate before resampling
            if self.apply_fades:
                fade_samples = int(self.soprano_sample_rate * self.fade_duration_ms / 1000)
                
                if len(audio_float32) > fade_samples:
                    fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)
                    audio_float32[:fade_samples] *= fade_in
                    
                    fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
                    audio_float32[-fade_samples:] *= fade_out
            
            # Determine target sample rate
            target_rate = self.output_sample_rate if self.output_sample_rate else self.soprano_sample_rate
            
            # Resample to target rate if needed
            if target_rate != self.soprano_sample_rate:
                audio_float32 = self._resample_audio(
                    audio_float32, 
                    self.soprano_sample_rate, 
                    target_rate
                )
                
                if self.debug:
                    resampled_duration = len(audio_float32) / target_rate
                    print(f"[SopranoEngine] Resampled: {len(audio_float32)} samples @ {target_rate}Hz, {resampled_duration:.2f}s")
            
            # Queue the complete resampled audio
            audio_bytes = audio_float32.tobytes()
            self.queue.put(audio_bytes)
            
            # Update duration tracking
            audio_duration = len(audio_float32) / target_rate
            self.audio_duration += audio_duration
            
            if self.debug:
                elapsed = time.time() - start_time
                rtf = audio_duration / elapsed if elapsed > 0 else 0
                print(f"[SopranoEngine] Total time: {elapsed:.2f}s, RTF: {rtf:.2f}x")
            
            return True
            
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            print(f"[SopranoEngine] Error: {e}")
            return False

    def set_voice(self, voice: Union[str, SopranoVoice]):
        """Updates the current voice."""
        self.current_voice = voice.name if isinstance(voice, SopranoVoice) else voice

    def set_voice_parameters(self, **voice_parameters):
        """Sets voice parameters."""
        if "chunk_size" in voice_parameters:
            self.chunk_size = voice_parameters["chunk_size"]
        if "output_sample_rate" in voice_parameters:
            self.output_sample_rate = voice_parameters["output_sample_rate"]

    def get_voices(self):
        """Returns available voices."""
        return [SopranoVoice("default")]

    def shutdown(self):
        """Cleanup."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__=="__main__":
    from RealtimeTTS import TextToAudioStream
    
    # Enable debug mode to see what's happening
    engine = SopranoEngine(debug=True)

    stream = TextToAudioStream(engine)
    
    # Save to file to check if artifacts are in the synthesis or playback
    stream.feed("Hello world! This is Soprano streaming in real-time.")
    stream.play(output_wavfile="soprano_realtimetts_test.wav")
    
    print("\nSaved to soprano_realtimetts_test.wav - check if artifacts are in the file itself")