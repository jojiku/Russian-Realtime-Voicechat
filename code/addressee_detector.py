# addressee_detector.py
import logging
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)

class AddresseeDetector:
    def __init__(self, model_path="models/addressee_detector/ru_ver/checkpoint-102", 
                 tokenizer_path="models/addressee_detector/rubert"):
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
        ("блин, этот кофе полное говно", 0),
        ("Люся, который сейчас час?", 1),
        ("бля, я так устал", 0),
        ("эй Люся, помоги мне", 1),
        ("код вообще пиздец какой-то", 0),
        ("Люся, код полный пиздец, помоги разобраться", 1),
        ("привет, как дела вообще?", 0),
        ("йо Люся", 1),
        ("привет народ / как дела пацаны", 0),
        ("ну ты сам знаешь как бывает", 0),
        ("я думаю тебе стоит", 0),
        ("что вообще блять происходит", 0),
        ("эй ассистент, как дела?", 1),
        ("ассистент, можешь помочь мне с этим разобраться?", 1),
        ("ладно ассистент", 1),
        ("компьютер, помоги мне", 1),
        ("эй компьютер", 1),
        ("тебе надо на это посмотреть", 1), 
        ("нам надо это починить", 0),
        ("кто-нибудь помогите", 0),
        ("кто-нибудь знает почему?", 0),
        ("может кто объяснит?", 0),
        ("куда я это положил?", 0),
        ("когда это было в последний раз?", 0),
        ("кто это сломал?", 0),
        ("какой вариант лучше?", 0),
        ("почему это дерьмо не работает?", 0),
        ("ты не поверишь", 0),
        ("нам нужно больше времени", 0),
        ("тут какая-то проблема", 0),
        ("что-то тут не так", 0),
        ("все нахуй сломалось", 0),
        ("покажи логи", 1),
        ("проверь базу данных", 1),
        ("запусти тесты снова", 1),
        ("останови сервер", 1),
        ("перезапусти все", 1),
        ("зачем я так сделал?", 0),
        ("о чем я вообще думал?", 0),
        ("как это вообще когда-то работало?", 0),
        ("куда я вообще с этим шел?", 0),
        ("когда это все сломалось?", 0),
        ("ну типа есть такая штука с...", 0),
        ("да, но проблема в том что", 0),
        ("окей, короче говоря", 0),
        ("я имею в виду, суть в том что", 0),
        ("ну очевидно же что мы не можем", 0),
        ("ты понимаешь о чем я?", 1),
        ("не могу поверить в это дерьмо", 0),
        ("вы че ребята издеваетесь", 0),
        ("остановите это", 0),
        ("почини этот бардак", 1),
        ("ладно, похуй, идем дальше", 0),
        ("похуй, сам разберусь", 0),
        ("ладно, допустим это работает", 0),
        ("ну ладно тогда", 0),
        ("прикольная история бро", 0),
        ("она понимает обращаюсь я к ней или нет", 0),
        ("да, она реально умная", 0),
        ("и она прикольная еще", 0),
        ("она просто классная", 0),
        ("да, она лучшая пока что", 0)
    ]

    test_texts = [x[0] for x in raw_data]
    test_labels = np.array([x[1] for x in raw_data])

    # 3. Inference Loop
    predictions = []
    print(f"Running inference on {len(test_texts)} cases...")

    for text in test_texts:
        pred = classifier.predict(text, 2)
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
