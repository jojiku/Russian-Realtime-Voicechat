# """
# Silero engine with sentence-level chunking for reduced latency.
# Splits text into sentences and synthesizes each separately.
# """

import os
import torch
import numpy as np
import pyaudio
import sys
import re
from queue import Queue
from typing import Union

from base_engine import BaseEngine


class SileroVoice:
    """Simple class to define a Silero speaker/voice."""
    def __init__(self, speaker_name: str, language: str, model_url: str, sample_rate: int):
        self.speaker_name = speaker_name
        self.language = language
        self.model_url = model_url
        self.sample_rate = sample_rate

    def __repr__(self):
        return f"SileroVoice(speaker='{self.speaker_name}', lang='{self.language}', rate={self.sample_rate})"


class SileroEngine(BaseEngine):
    """
    Sentence-chunked Silero TTS engine.
    Splits text into sentences and synthesizes each separately for lower latency.
    """
    def __init__(
        self,
        voice: Union[str, SileroVoice] = 'kseniya',
        language: str = 'ru',
        model_name: str = 'v5_1_ru',
        sample_rate: int = 48000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        thread_count: int = 4,
        chunk_size: int = 4096,
        local_file_path: str = 'models/silero/model.pt',
        prosody_rate: str = 'fast'
    ):
        self.engine_name = "silero"
        self.model = None
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.local_file_path = local_file_path
        self.prosody_rate = prosody_rate
        
        torch.set_num_threads(thread_count)
        
        if isinstance(voice, str):
            self.current_voice_name = voice
            self.current_language = language
        else:
            self.current_voice_name = voice.speaker_name
            self.current_language = voice.language
            self.sample_rate = voice.sample_rate

        self.model_url = f'https://models.silero.ai/models/tts/{self.current_language}/{model_name}.pt'
        self.load_model()
        
    def load_model(self):
        """Downloads and loads the Silero model."""
        import time
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)
        
        # Download if needed
        if not os.path.isfile(self.local_file_path):
            print(f"[SileroEngine] Downloading model from {self.model_url}...")
            temp_path = self.local_file_path + '.download'
            
            try:
                # Download to temporary file first
                torch.hub.download_url_to_file(self.model_url, temp_path)
                
                # Verify the download completed
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"Download failed - temp file not created: {temp_path}")
                
                # Move to final location atomically
                os.replace(temp_path, self.local_file_path)
                print(f"[SileroEngine] Model download complete.")
                
            except Exception as e:
                # Clean up partial downloads
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"Model download failed: {e}")
        
        # Wait for any partial downloads to clear (safety check)
        max_wait = 10
        waited = 0
        while any(f.endswith('.partial') for f in os.listdir(os.path.dirname(self.local_file_path))):
            if waited >= max_wait:
                raise RuntimeError("Timed out waiting for partial downloads to complete")
            time.sleep(1)
            waited += 1
        
        # Load the model
        try:
            print(f"[SileroEngine] Loading model from {self.local_file_path}...")
            self.model = torch.package.PackageImporter(self.local_file_path).load_pickle("tts_models", "model")
            self.model.to(self.device)
            print(f"[SileroEngine] Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"[SileroEngine] Error loading model: {e}")
            # Clean up corrupted file
            if os.path.exists(self.local_file_path):
                os.remove(self.local_file_path)
            self.model = None
            raise

    def get_stream_info(self):
        """Returns PyAudio stream format: 32-bit float, 1 channel, sample rate."""
        return pyaudio.paFloat32, 1, self.sample_rate

    def set_voice(self, voice: Union[str, SileroVoice]):
        """Sets the speaker name."""
        if isinstance(voice, SileroVoice):
            self.current_voice_name = voice.speaker_name
        elif isinstance(voice, str):
            self.current_voice_name = voice
        
        print(f"[SileroEngine] Voice (speaker) set to: {self.current_voice_name}")

    def set_prosody_rate(self, rate: str):
        """Sets the prosody rate for speech synthesis."""
        self.prosody_rate = rate
        print(f"[SileroEngine] Prosody rate set to: {self.prosody_rate}")

    def _wrap_with_prosody(self, text: str) -> str:
        """Wraps text in SSML with prosody rate if set."""
        if self.prosody_rate:
            return f'<speak><prosody rate="{self.prosody_rate}">{text}</prosody></speak>'
        return text

    def _split_into_sentences(self, text: str) -> list:
        """
        Splits text into sentences for individual synthesis.
        Uses regex to split on sentence boundaries while preserving punctuation.
        """
        # Russian sentence endings
        sentence_delimiters = r'[.!?â€¦ã€‚]+[\s]+'
        
        # Split but keep the delimiter
        parts = re.split(f'({sentence_delimiters})', text)
        
        sentences = []
        for i in range(0, len(parts)-1, 2):
            sentence = parts[i]
            if i+1 < len(parts):
                sentence += parts[i+1].rstrip()  # Add delimiter, remove trailing space
            
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        
        # Handle last part if it doesn't end with delimiter
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        
        return sentences if sentences else [text.strip()]

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text by splitting into sentences and processing each separately.
        This dramatically reduces time to first audio chunk.
        """
        super().synthesize(text)

        if self.model is None:
            sys.stderr.write("[SileroEngine] Model is not loaded. Cannot synthesize.\n")
            return False

        if self.stop_synthesis_event.is_set():
            self.stop_synthesis_event.clear()

        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            for idx, sentence in enumerate(sentences):
                # Check for stop between sentences
                if self.stop_synthesis_event.is_set():
                    self.queue.put(None)
                    return False
                
                # Synthesize this sentence
                ssml_text = self._wrap_with_prosody(sentence)
                
                
                audio_tensor = self.model.apply_tts(
                    ssml_text=ssml_text,
                    speaker=self.current_voice_name,
                    sample_rate=self.sample_rate
                )

                audio_numpy = audio_tensor.cpu().numpy().astype(np.float32)
                
                # Update duration tracking
                audio_length_in_seconds = len(audio_numpy) / self.sample_rate
                self.audio_duration += audio_length_in_seconds
                
                # Stream this sentence in chunks
                total_samples = len(audio_numpy)
                for i in range(0, total_samples, self.chunk_size):
                    # Check for stop during chunking
                    if self.stop_synthesis_event.is_set():
                        print(f"[SileroEngine] Synthesis stopped during chunking of sentence {idx+1}")
                        self.queue.put(None)
                        return False
                    
                    chunk = audio_numpy[i:i + self.chunk_size]

                    self.queue.put(chunk.tobytes())
            
            # Signal end of stream
            self.queue.put(None)
            return True

        except Exception as e:
            sys.stderr.write(f"[SileroEngine] Error during synthesis: {e}\n")
            self.queue.put(None)
            return False

    def shutdown(self):
        """Release TTS engine and GPU resources."""
        print("[SileroEngine] Shutting down...")
        
        # Stop any ongoing synthesis
        if hasattr(self, 'stop_synthesis_event') and self.stop_synthesis_event: 
            self.stop_synthesis_event.set()
        
        # Delete the model
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Move model to CPU first to free GPU memory more reliably
                self.model.to('cpu')
                del self.model
                self.model = None
                print("[SileroEngine] Model deleted.")
            except Exception as e:
                print(f"[SileroEngine] Error deleting model: {e}")
        
        # Force CUDA cleanup
        try: 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[SileroEngine] CUDA cache cleared.")
        except Exception as e: 
            print(f"[SileroEngine] Error clearing CUDA cache: {e}")
        
        print("[SileroEngine] Shutdown complete.")

    def __del__(self):
        """Destructor - attempt cleanup if shutdown wasn't called."""
        try:
            if hasattr(self, 'model') and self.model is not None: 
                self.shutdown()
        except Exception: 
            pass
        


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    from silero_engine import SileroEngine
    
    test_text = "Привет как дела?"
    engine = SileroEngine(chunk_size=2048)
    engine.set_prosody_rate('medium')
    
    print("Creating stream...")
    from RealtimeTTS import TextToAudioStream
    stream = TextToAudioStream(
        engine, 
        playout_chunk_size=4096,
        frames_per_buffer=2048
    )   
    
    print("Starting playback...")
    stream.feed(test_text).play(log_synthesized_text=True)
