# addressee_detector.py
import logging
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)

class AddresseeDetector:
    def __init__(self, model_path="models/addressee_detector/en_ver/finetuned_adressee/checkpoint-114", 
                 tokenizer_path="models/addressee_detector/en_ver/tokenizer/"):
        """The brain police. Decides if you are worthy of my attention."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_loader = AutoTokenizer.from_pretrained(tokenizer_path)
        
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_path, 
                tokenizer=tokenizer_loader,
                device=0 if self.device == "cuda" else -1,
                top_k=None
            )
            logger.info("AddresseeDetector loaded.")
        except Exception as e:
            logger.error(f"Failed to load addressee model: {e}")
            self.classifier = None

        # State tracking
        self.last_interaction_time = 0
        
        # Configuration
        self.WAKE_WORDS = ["lucy", "computer", "assistant"]
        self.BASE_THRESHOLD = 0.55  # Lowered for active conversations
        self.CONVERSATION_WINDOW = 10.0  # Extended to 10 seconds
        self.CONTEXT_BOOST = 0.35  # Increased boost during active conversation

    def predict(self, text):
        """Returns probability (0.0 - 1.0) that text is addressed to Lucy."""
        if not self.classifier:
            return 1.0
        
        results = self.classifier(text)[0]
        
        for res in results:
            if res['label'] == 'LABEL_1':
                return res['score']
            if res['label'] == 'LABEL_0':
                return 1.0 - res['score']
        
        return 0.0

    def should_reply(self, text, time_since_ai_spoke):
        """The Master Logic. Simple version."""
        text_lower = text.lower().strip()
        
        # Wake word = instant yes
        for word in self.WAKE_WORDS:
            if word in text_lower:
                logger.info(f"Wake word detected: '{word}'")
                self.last_interaction_time = time.time()
                return True
        
        # Get base prediction
        base_score = self.predict(text)
        
        # Apply context boost if we're in active conversation
        final_score = base_score
        if time_since_ai_spoke < self.CONVERSATION_WINDOW:
            decay_factor = 1.0 - (time_since_ai_spoke / self.CONVERSATION_WINDOW)
            final_score += self.CONTEXT_BOOST * decay_factor
        
        final_score = min(1.0, final_score)
        
        logger.info(f"Analysis: Base={base_score:.2f} | Final={final_score:.2f} | Threshold={self.BASE_THRESHOLD}")
        
        # Single threshold decision
        should_reply = final_score >= self.BASE_THRESHOLD
        
        if should_reply:
            self.last_interaction_time = time.time()
        
        return should_reply

if __name__=="__main__":
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    classifier = AddresseeDetector()

    # 2. Define Test Data (Extracted from your list)
    # Format: (text, expected_label)
    raw_data = [
        ("man, this coffee is total trash", 0),
        ("Lucy, what time is it right now?", 1),
        ("damn, I'm so tired", 0),
        ("hey Lucy, help me out", 1),
        ("the code is just a total fucking mess", 0),
        ("Lucy, the code is a total mess, help me out", 1),
        ("hey man, how's it going anyway?", 0),
        ("yo Lucy", 1),
        ("hey guys / what's up fellas", 0),
        ("well, you know how it is", 0),
        ("I think you should", 0),
        ("what the fuck is even going on", 0),
        ("hey assistant, how's it going?", 1),
        ("assistant, can you help me figure this out?", 1),
        ("okay assistant", 1),
        ("computer, help me", 1),
        ("hey computer", 1),
        ("you need to see this", 1), 
        ("we need to fix this", 0),
        ("somebody help me", 0),
        ("does anyone know why?", 0),
        ("can someone explain?", 0),
        ("where did I put that?", 0),
        ("when was the last time?", 0),
        ("who broke this?", 0),
        ("which option is better?", 0),
        ("why isn't this crap working?", 0),
        ("you won't believe this", 0),
        ("we're gonna need more time", 0),
        ("there's some kind of problem here", 0),
        ("something's not right here", 0),
        ("everything is fucking broken", 0),
        ("show logs", 1),
        ("check the database", 1),
        ("run the tests again", 1),
        ("stop the server", 1),
        ("restart everything", 1),
        ("why did I do it like that?", 0),
        ("what was I even thinking?", 0),
        ("how did this ever even work?", 0),
        ("where was I even going with this?", 0),
        ("when did it all break?", 0),
        ("well, like, there's this thing with...", 0),
        ("yeah, but the problem is that", 0),
        ("okay, well, long story short", 0),
        ("I mean, the point here is", 0),
        ("well obviously we can't", 0),
        ("do you know what I mean?", 1),
        ("I can't believe this shit", 0),
        ("you guys must be kidding me", 0),
        ("make it stop", 0),
        ("fix this mess", 1),
        ("anyway, whatever, let's move on", 0),
        ("whatever, I'll figure it out myself", 0),
        ("fine, let's assume it works", 0),
        ("alright then", 0),
        ("cool story bro", 0),
        ("she understands if I'm approaching her or not", 0),
        ("yeah, she's actually really smart", 0),
        ("and she's funny, too", 0),
        ("she's just great", 0),
        ("yeah, she's the best one so far", 0)
    ]

    test_texts = [x[0] for x in raw_data]
    test_labels = np.array([x[1] for x in raw_data])

    # 3. Inference Loop
    predictions = []
    print(f"Running inference on {len(test_texts)} cases...")

    for text in test_texts:
        pred = classifier.predict(text)
        pred = np.round(pred)
        predictions.append(pred)

    # 4. Metrics Calculation
    predictions = np.array(predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    # 5. Print Report
    print("\n" + "="*50)
    print("ADDRESSEE DETECTOR EVALUATION")
    print("="*50)
    print(f"Accuracy:           {accuracy*100:.2f}%")
    print(f"Precision:          {precision*100:.2f}%")
    print(f"Recall:             {recall*100:.2f}%")
    print(f"F1-Score:           {f1*100:.2f}%")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"Predicted [0] | Actual [0]: {cm[0][0]} (TN)")
    print(f"Predicted [1] | Actual [0]: {cm[0][1]} (FP)")
    print(f"Predicted [0] | Actual [1]: {cm[1][0]} (FN)")
    print(f"Predicted [1] | Actual [1]: {cm[1][1]} (TP)")
    print("="*50)

    # 6. Show Failures
    print("\nFAILED TEST CASES:")
    has_failures = False
    for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, predictions)):
        if true_label != pred_label:
            has_failures = True
            t_name = "ADDRESSED" if true_label == 1 else "NOT_ADDRESSED"
            p_name = "ADDRESSED" if pred_label == 1 else "NOT_ADDRESSED"
            print(f"✖ '{text}'\n  Expected: {t_name} | Got: {p_name}\n")

    if not has_failures:
        print("All tests passed perfectly!")
