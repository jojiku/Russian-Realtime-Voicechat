# Adressee_detector.py
import logging
import time
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
logger = logging.getLogger(__name__)

class AddresseeDetector:
    def __init__(self, model_path="models/addressee_detector/checkpoint-408", tokenizer_path = "models/addressee_detector/tokenizer"):
        """
        The brain police. Decides if you are worthy of my attention.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_loader = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_path, 
                tokenizer=tokenizer_loader,
                device=0 if self.device == "cuda" else -1,
                top_k=None # Return all scores
            )
            logger.info("AddresseeDetector loaded.")
        except Exception as e:
            logger.error(f"Failed to load addressee model: {e}")
            self.classifier = None

        # State tracking
        self.last_interaction_time = 0
        self.conversation_momentum = 0.0 # 0.0 to 1.0
        
        # Configuration
        self.WAKE_WORDS = ["lucy", "computer", "assistant"]
        self.CONFIDENCE_THRESHOLD_HIGH = 0.85
        self.CONFIDENCE_THRESHOLD_LOW = 0.35 # Gray zone floor
        
        # Time constants
        self.MOMENTUM_DECAY_SECONDS = 30.0
        self.HOT_CONVERSATION_WINDOW = 10.0 # Seconds where I'm paying full attention

    def predict(self, text, time_since_last_response=None):
        """
        Returns probability (0.0 - 1.0) that text is addressed to Lucy.
        """
        if not self.classifier:
            return 1.0 # Fail open if model missing

        # 1. Base Model Prediction
        results = self.classifier(text)[0] # List of dicts [{'label': 'LABEL_0', 'score': 0.9}, ...]
        
        # Extract score for LABEL_1 (Addressing Lucy)
        label_1_score = 0.0
        for res in results:
            if res['label'] == 'LABEL_1':
                label_1_score = res['score']
                break
            if res['label'] == 'LABEL_0':
                label_1_score = 1.0 - res['score']
                
        return label_1_score

    def should_reply(self, text, time_since_ai_spoke):
        """
        The Master Logic. Applies heuristics to the base prediction.
        """
        text_lower = text.lower().strip()
        now = time.time()
        
        # --- Rule 15: Explicit Wake Word Override ---
        # If you say my name, I listen. Period.
        for word in self.WAKE_WORDS:
            if word in text_lower:
                logger.info(f"👀👂 Explicit wake word detected: '{word}'")
                self._update_momentum(hit=True)
                return True

        # --- Rule 3: Utterance Length Penalty ---
        # Don't trigger on "okay", "shit", "nice" unless we are DEEP in conversation
        word_count = len(text.split())
        length_penalty = 0.0
        if word_count <= 2:
            length_penalty = 0.25 # Heavy penalty for short grunts
        elif word_count <= 4:
            length_penalty = 0.1

        # --- Rule 1 & 4: Context & Time Boost ---
        # If I just spoke, you are likely replying to me.
        # Decay this boost linearly over HOT_CONVERSATION_WINDOW seconds.
        time_boost = 0.0
        if time_since_ai_spoke < self.HOT_CONVERSATION_WINDOW:
            # e.g., 0s elapsed -> 1.0 factor. 10s elapsed -> 0.0 factor.
            decay_factor = 1.0 - (time_since_ai_spoke / self.HOT_CONVERSATION_WINDOW)
            time_boost = 0.3 * decay_factor # Max 0.3 boost

        # --- Rule 1: Momentum ---
        # If we've been chatting, I'm more likely to listen.
        # Decay momentum if it's been a while since you addressed me.
        time_since_last_interaction = now - self.last_interaction_time
        if time_since_last_interaction > self.MOMENTUM_DECAY_SECONDS:
            self.conversation_momentum = 0.0
        
        momentum_boost = self.conversation_momentum * 0.15 # Max 0.15 boost

        # --- Get Base Score ---
        base_score = self.predict(text)

        # --- Calculate Final Score ---
        final_score = base_score + time_boost + momentum_boost - length_penalty
        
        # Clamp
        final_score = max(0.0, min(1.0, final_score))

        logger.info(f"👀 Analysis: Base={base_score:.2f} | TimeBoost={time_boost:.2f} | Momentum={momentum_boost:.2f} | LenPen={length_penalty:.2f} | FINAL={final_score:.2f}")

        # --- Decision Time (Rule 2: Thresholds) ---
        should_reply = False
        
        if final_score > self.CONFIDENCE_THRESHOLD_HIGH:
            should_reply = True
        elif final_score > self.CONFIDENCE_THRESHOLD_LOW:
            # Gray zone logic: If we are already in a "hot" flow, lean yes.
            if time_since_ai_spoke < 5.0 or self.conversation_momentum > 0.5:
                should_reply = True
                logger.info("👀 Gray zone ACCEPT (Context/Momentum saved you)")
            else:
                should_reply = False
                logger.info("👀 Gray zone REJECT (Not enough context)")
        else:
            should_reply = False

        if should_reply:
            self._update_momentum(hit=True)
        else:
            self._update_momentum(hit=False)
        logger.info("should I talk score:", final_score)
        return should_reply#, final_score

    def _update_momentum(self, hit=True):
        self.last_interaction_time = time.time()
        if hit:
            # Boost momentum, cap at 1.0
            self.conversation_momentum = min(1.0, self.conversation_momentum + 0.3)
        else:
            # Decay momentum on ignore
            self.conversation_momentum = max(0.0, self.conversation_momentum - 0.2)

if __name__=="__main__":
    addressee_detector = AddresseeDetector()
    print(addressee_detector.should_reply("yo Lucy u playin safe today?", 2)) # Should give True
    print(addressee_detector.should_reply("She knows when i speak not to her", 2)) # Should give False
    print(addressee_detector.should_reply("she is a smart girl", 2)) # Should give False
