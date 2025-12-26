"""
Latency tracking system for conversational bot.
Measures: Whisper STT, LLM inference, TTS synthesis, and total pipeline latency.

Also includes a streaming-enabled Silero engine fix.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Container for a single conversation turn's latency measurements."""
    
    # Timestamps
    user_speech_end: float = 0.0          # When user stopped speaking
    stt_complete: float = 0.0             # When transcription finished
    llm_first_token: float = 0.0          # When LLM produced first token
    llm_context_ready: float = 0.0        # When quick answer context found
    llm_complete: float = 0.0             # When LLM finished generating
    tts_quick_start: float = 0.0          # When TTS quick synthesis started
    tts_first_chunk: float = 0.0          # When first audio chunk ready
    tts_quick_complete: float = 0.0       # When quick TTS finished
    tts_final_complete: float = 0.0       # When final TTS finished (if applicable)
    
    # Derived latencies (in milliseconds)
    stt_latency: float = 0.0              # Time for speech-to-text
    llm_ttft: float = 0.0                 # Time to first token
    llm_context_latency: float = 0.0      # Time to get quick answer
    llm_total_latency: float = 0.0        # Total LLM generation time
    tts_inference_time: float = 0.0       # Time from text to first audio
    tts_quick_total: float = 0.0          # Total quick TTS time
    tts_final_total: float = 0.0          # Total final TTS time
    
    # The big one
    total_pipeline_latency: float = 0.0   # User stop -> First audio chunk
    
    # Context
    generation_id: int = 0
    quick_answer_length: int = 0
    final_answer_length: int = 0
    
    def calculate_latencies(self):
        """Calculate all derived latency metrics."""
        if self.stt_complete > 0 and self.user_speech_end > 0:
            self.stt_latency = (self.stt_complete - self.user_speech_end) * 1000
        
        if self.llm_first_token > 0 and self.stt_complete > 0:
            self.llm_ttft = (self.llm_first_token - self.stt_complete) * 1000
        
        if self.llm_context_ready > 0 and self.stt_complete > 0:
            self.llm_context_latency = (self.llm_context_ready - self.stt_complete) * 1000
        
        if self.llm_complete > 0 and self.stt_complete > 0:
            self.llm_total_latency = (self.llm_complete - self.stt_complete) * 1000
        
        if self.tts_first_chunk > 0 and self.tts_quick_start > 0:
            self.tts_inference_time = (self.tts_first_chunk - self.tts_quick_start) * 1000
        
        if self.tts_quick_complete > 0 and self.tts_quick_start > 0:
            self.tts_quick_total = (self.tts_quick_complete - self.tts_quick_start) * 1000
        
        if self.tts_final_complete > 0 and self.tts_final_complete > self.tts_quick_complete:
            self.tts_final_total = (self.tts_final_complete - self.tts_quick_complete) * 1000
        
        # Total: from user speech end to first audio chunk
        if self.tts_first_chunk > 0 and self.user_speech_end > 0:
            self.total_pipeline_latency = (self.tts_first_chunk - self.user_speech_end) * 1000
    
    def log_summary(self):
        """Log a formatted summary of all latencies."""
        self.calculate_latencies()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"LATENCY REPORT - Generation {self.generation_id}")
        logger.info(f"{'='*60}")
        logger.info(f"🎤 STT Latency:           {self.stt_latency:>8.2f} ms")
        logger.info(f"🧠 LLM TTFT:              {self.llm_ttft:>8.2f} ms")
        logger.info(f"🧠 LLM Context Ready:     {self.llm_context_latency:>8.2f} ms")
        logger.info(f"🧠 LLM Total:             {self.llm_total_latency:>8.2f} ms")
        logger.info(f"💬 TTS Inference:         {self.tts_inference_time:>8.2f} ms")
        logger.info(f"💬 TTS Quick Total:       {self.tts_quick_total:>8.2f} ms")
        if self.tts_final_total > 0:
            logger.info(f"💬 TTS Final Total:       {self.tts_final_total:>8.2f} ms")
        logger.info(f"{'-'*60}")
        logger.info(f"⚡ TOTAL PIPELINE:        {self.total_pipeline_latency:>8.2f} ms")
        logger.info(f"{'='*60}\n")
        
        # Breakdown percentages
        if self.total_pipeline_latency > 0:
            stt_pct = (self.stt_latency / self.total_pipeline_latency) * 100
            llm_pct = (self.llm_context_latency / self.total_pipeline_latency) * 100
            tts_pct = (self.tts_inference_time / self.total_pipeline_latency) * 100
            
            logger.info(f"Breakdown: STT={stt_pct:.1f}%, LLM={llm_pct:.1f}%, TTS={tts_pct:.1f}%")


class LatencyTracker:
    """
    Tracks latency metrics across the entire conversational pipeline.
    Use this as a singleton attached to your SpeechPipelineManager.
    """
    
    def __init__(self, history_size: int = 10):
        self.current: Optional[LatencyMetrics] = None
        self.history: deque = deque(maxlen=history_size)
        self.history_size = history_size
    
    def start_new_turn(self, generation_id: int):
        """Start tracking a new conversation turn."""
        if self.current:
            # Finalize previous turn
            self.current.calculate_latencies()
            self.history.append(self.current)
        
        self.current = LatencyMetrics(generation_id=generation_id)
        logger.debug(f"📊 Started latency tracking for Gen {generation_id}")
    
    def mark_user_speech_end(self):
        """Mark when user stopped speaking (before_final callback)."""
        if self.current:
            self.current.user_speech_end = time.time()
            logger.debug(f"📊 [Gen {self.current.generation_id}] User speech ended")
    
    def mark_stt_complete(self):
        """Mark when STT transcription completed (on_final callback)."""
        if self.current:
            self.current.stt_complete = time.time()
            if self.current.user_speech_end > 0:
                latency = (self.current.stt_complete - self.current.user_speech_end) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] STT complete: {latency:.2f}ms")
    
    def mark_llm_first_token(self):
        """Mark when LLM produced its first token."""
        if self.current:
            self.current.llm_first_token = time.time()
            if self.current.stt_complete > 0:
                ttft = (self.current.llm_first_token - self.current.stt_complete) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] LLM TTFT: {ttft:.2f}ms")
    
    def mark_llm_context_ready(self, quick_answer_length: int):
        """Mark when LLM produced enough context for quick answer."""
        if self.current:
            self.current.llm_context_ready = time.time()
            self.current.quick_answer_length = quick_answer_length
            if self.current.stt_complete > 0:
                latency = (self.current.llm_context_ready - self.current.stt_complete) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] LLM context ready: {latency:.2f}ms ({quick_answer_length} chars)")
    
    def mark_llm_complete(self, final_answer_length: int):
        """Mark when LLM finished generating."""
        if self.current:
            self.current.llm_complete = time.time()
            self.current.final_answer_length = final_answer_length
            if self.current.stt_complete > 0:
                latency = (self.current.llm_complete - self.current.stt_complete) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] LLM complete: {latency:.2f}ms ({final_answer_length} chars)")
    
    def mark_tts_quick_start(self):
        """Mark when TTS quick synthesis started."""
        if self.current:
            self.current.tts_quick_start = time.time()
            logger.debug(f"📊 [Gen {self.current.generation_id}] TTS quick started")
    
    def mark_tts_first_chunk(self):
        """Mark when first TTS audio chunk was generated (CRITICAL METRIC)."""
        if self.current:
            self.current.tts_first_chunk = time.time()
            
            # Calculate and log key latencies
            if self.current.tts_quick_start > 0:
                tts_inference = (self.current.tts_first_chunk - self.current.tts_quick_start) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] TTS first chunk: {tts_inference:.2f}ms")
            
            if self.current.user_speech_end > 0:
                total = (self.current.tts_first_chunk - self.current.user_speech_end) * 1000
                logger.info(f"📊 [Gen {self.current.generation_id}] ⚡ TOTAL PIPELINE: {total:.2f}ms ⚡")
    
    def mark_tts_quick_complete(self):
        """Mark when TTS quick synthesis completed."""
        if self.current:
            self.current.tts_quick_complete = time.time()
            if self.current.tts_quick_start > 0:
                latency = (self.current.tts_quick_complete - self.current.tts_quick_start) * 1000
                logger.debug(f"📊 [Gen {self.current.generation_id}] TTS quick complete: {latency:.2f}ms")
    
    def mark_tts_final_complete(self):
        """Mark when TTS final synthesis completed."""
        if self.current:
            self.current.tts_final_complete = time.time()
            if self.current.tts_quick_complete > 0:
                latency = (self.current.tts_final_complete - self.current.tts_quick_complete) * 1000
                logger.debug(f"📊 [Gen {self.current.generation_id}] TTS final complete: {latency:.2f}ms")
    
    def finalize_turn(self):
        """Finalize current turn and log summary."""
        if self.current:
            self.current.log_summary()
            self.history.append(self.current)
            self.current = None
    
    def get_average_latencies(self) -> Dict[str, float]:
        """Get average latencies across recent history."""
        if not self.history:
            return {}
        
        metrics = {
            'stt_latency': [],
            'llm_ttft': [],
            'llm_context_latency': [],
            'tts_inference_time': [],
            'total_pipeline_latency': []
        }
        
        for turn in self.history:
            turn.calculate_latencies()
            for key in metrics:
                value = getattr(turn, key, 0)
                if value > 0:
                    metrics[key].append(value)
        
        averages = {}
        for key, values in metrics.items():
            if values:
                averages[f'avg_{key}'] = sum(values) / len(values)
                averages[f'min_{key}'] = min(values)
                averages[f'max_{key}'] = max(values)
        
        return averages
    
    def log_statistics(self):
        """Log aggregated statistics from history."""
        stats = self.get_average_latencies()
        if not stats:
            logger.info("No latency statistics available yet")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"LATENCY STATISTICS (last {len(self.history)} turns)")
        logger.info(f"{'='*60}")
        
        for metric in ['stt_latency', 'llm_ttft', 'llm_context_latency', 'tts_inference_time', 'total_pipeline_latency']:
            avg_key = f'avg_{metric}'
            min_key = f'min_{metric}'
            max_key = f'max_{metric}'
            
            if avg_key in stats:
                logger.info(f"{metric:25s}: avg={stats[avg_key]:>7.2f}ms  min={stats[min_key]:>7.2f}ms  max={stats[max_key]:>7.2f}ms")
        
        logger.info(f"{'='*60}\n")