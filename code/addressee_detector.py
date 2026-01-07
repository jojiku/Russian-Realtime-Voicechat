# addressee_detector.py
import logging
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)
model_dir = "Silxxor/Russian-Addressee-detector"

class AddresseeDetector: 
    def __init__(self):
        """The brain police.  Decides if you are worthy of my attention."""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_dir, 
                tokenizer=AutoTokenizer.from_pretrained(model_dir),
                device=0 if self.device == "cuda" else -1,
                top_k=None
            )
            logger.info("AddresseeDetector loaded.")
        except Exception as e:
            logger.error(f"Failed to load addressee model: {e}")
            self.classifier = None

        self.last_interaction_time = 0
        
        self.WAKE_WORDS = ["люси", "компьютер", "ассистент", "lucy", "computer", "assistant"]
        self.CONVERSATION_WINDOW = 10.0
        
        self.HIGH_THRESHOLD = 0.75
        self.LOW_THRESHOLD = 0.25

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
        """
        Simple logic: 
        1. Wake word = YES
        2. Within 5s of last AI response = YES (unless model is VERY sure it's not for us)
        3. Cold start = trust the model
        """
        text_lower = text.lower().strip()
        
        # === WAKE WORDS:  Always yes ===
        for word in self.WAKE_WORDS: 
            if word.lower() in text_lower:
                return True
        
        base_score = self.predict(text)
        in_conversation = time_since_ai_spoke < 5.0
        
        if in_conversation:
            # Recently talked = assume they're still talking to us
            # UNLESS model is very confident it's NOT for us (< 0.15)
            if base_score < 0.15:
                logger.info(f"[ACTIVE] score={base_score:.2f} < 0.15 -> NO (model very confident)")
                return False
            else: 
                logger.info(f"[ACTIVE] score={base_score:.2f} within 5s -> YES")
                return True
        else: 
            # Cold start = need model confidence
            threshold = 0.5
            result = base_score >= threshold
            logger.info(f"[COLD] score={base_score:.2f} >= {threshold} -> {result}")
            return result
    
    def shutdown(self):
        """Release GPU memory and cleanup resources."""
        logger.info("🧠 AddresseeDetector:  Shutting down...")
        
        if self.classifier is not None:
            # Delete the pipeline and its underlying model
            del self.classifier
            self.classifier = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧠 AddresseeDetector: Shutdown complete.")

    def __del__(self):
        """Destructor - attempt cleanup if shutdown wasn't called."""
        try: 
            if hasattr(self, 'classifier') and self.classifier is not None: 
                self.shutdown()
        except Exception: 
            pass


if __name__=="__main__":
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    classifier = AddresseeDetector()

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

    predictions = []
    print(f"Running inference on {len(test_texts)} cases...")

    for text in test_texts:
        pred = classifier.predict(text)
        pred = np.round(pred)
        predictions.append(pred)

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

    #=================================================================
    print("\n" + "="*70)
    print("THREE-TIER LOGIC ANALYSIS")
    print("="*70)
    print(f"HIGH_THRESHOLD (Tier2 YES): >= {classifier.HIGH_THRESHOLD}")
    print(f"LOW_THRESHOLD (Tier2 NO):   <= {classifier.LOW_THRESHOLD}")
    print(f"AMBIGUOUS ZONE (Tier3):     {classifier.LOW_THRESHOLD} < score < {classifier.HIGH_THRESHOLD}")
    print("="*70 + "\n")
    
    # Test phrases with their expected tier behavior
    test_phrases = [
        # (text, description)
        ("Люси, привет!", "Wake word - should be TIER1"),
        ("эй компьютер", "Wake word - should be TIER1"),
        ("покажи логи", "Clear command - likely TIER2-YES or high TIER3"),
        ("перезапусти сервер", "Clear command"),
        ("блин, этот кофе говно", "Self-talk - should be TIER2-NO"),
        ("бля, я устал", "Self-talk - should be TIER2-NO"),
        ("она такая умная", "Talking ABOUT AI - should be TIER2-NO"),
        ("интересно.. .", "Ambiguous - should be TIER3"),
        ("да", "Ambiguous - should be TIER3"),
        ("нет, не то", "Ambiguous - should be TIER3"),
        ("хм, понятно", "Ambiguous - should be TIER3"),
        ("продолжай", "Ambiguous command - likely TIER3"),
        ("ты понимаешь? ", "Ambiguous question - likely TIER3"),
        ("как у тебя дела?", "Conversational - likely TIER3"),
        ("а что если попробовать?", "Follow-up - likely TIER3"),
    ]
    
    print(f"{'Text':<45} {'Score': >6} {'Tier':<12} {'Cold': >6} {'@1s':>6} {'@5s':>6}")
    print("-"*85)
    
    for text, desc in test_phrases: 
        score = classifier.predict(text)
        
        # Determine tier
        text_lower = text.lower()
        is_wake = any(w in text_lower for w in classifier.WAKE_WORDS)
        
        if is_wake:
            tier = "TIER1-WAKE"
        elif score >= classifier.HIGH_THRESHOLD:
            tier = "TIER2-YES"
        elif score <= classifier.LOW_THRESHOLD:
            tier = "TIER2-NO"
        else:
            tier = "TIER3-AMB"
        
        # Test at different time contexts
        cold_result = classifier.should_reply(text, 100.0)
        at_1s = classifier.should_reply(text, 1.0)
        at_5s = classifier.should_reply(text, 5.0)
        
        cold_str = "✅" if cold_result else "❌"
        at_1s_str = "✅" if at_1s else "❌"
        at_5s_str = "✅" if at_5s else "❌"
        
        print(f"{text:<45} {score:>6.2f} {tier:<12} {cold_str:>6} {at_1s_str:>6} {at_5s_str: >6}")
    
    print("\n" + "="*70)
    print("Legend:  Cold = no recent AI speech | @1s/@5s = seconds since AI spoke")
    print("="*70)


    
    print("\n" + "="*70)
    print("AMBIGUOUS PHRASES:  CONTEXT SENSITIVITY TEST")
    print("="*70)
    print("These phrases are INTENTIONALLY ambiguous.  The model can't know.")
    print("Context (time since AI spoke) should be the deciding factor.")
    print("="*70 + "\n")
    
    ambiguous = [
        "интересно...",
        "да",
        "нет",
        "понятно",
        "хорошо",
        "окей",
        "продолжай",
        "дальше",
        "а потом? ",
        "и что? ",
        "серьезно?",
        "ого",
        "ну давай",
        "попробуй",
    ]
    
    time_points = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 15.0]
    
    print(f"{'Phrase':<20}", end="")
    for t in time_points:
        print(f"{t:>5}s", end=" ")
    print(f"{'Score':>7}")
    print("-" * 75)
    
    for phrase in ambiguous:
        score = classifier.predict(phrase)
        print(f"{phrase:<20}", end="")
        
        for t in time_points:
            result = classifier.should_reply(phrase, t)
            symbol = "✅" if result else "·"  # Using dot for cleaner view
            print(f"{symbol: >5}", end=" ")
        
        print(f"{score:>7.2f}")
    
    print("\n" + "-"*75)
    print("✅ = will reply | · = will ignore")
    print(f"Conversation window: {classifier.CONVERSATION_WINDOW}s")


    
    print("\n" + "="*70)
    print("REALISTIC CONVERSATION SIMULATION")
    print("="*70 + "\n")
    
    last_ai_spoke = 0
    
    conversation = [
        # (user_says, delay_seconds, expected_response, scenario_note)
        ("Люси, привет!", 0, True, "Wake word starts convo"),
        ("как дела?", 1.5, True, "Quick follow-up"),
        ("а что ты умеешь?", 2.0, True, "Continuing conversation"),
        ("интересно", 1.0, True, "Short reaction - AMBIGUOUS but recent"),
        ("покажи пример", 1.5, True, "Command in active convo"),
        ("ага, понял", 2.0, True, "Confirmation"),
        # User turns away to talk to friend
        ("Вась, глянь что она умеет", 5.0, False, "Talking to friend Vasya"),
        ("да, прикольная штука", 3.0, False, "Still talking to friend"),
        ("ну ладно, пойдем", 4.0, False, "Leaving with friend"),
        # Long pause, comes back
        ("Люси, еще вопрос", 20.0, True, "Wake word after long pause"),
        ("спасибо, все понятно", 2.0, True, "Closing in active convo"),
    ]
    
    for user_text, delay, expected, note in conversation:
        if delay > 0:
            print(f"    ... {delay}s pause ...")
            time.sleep(delay)
        
        if last_ai_spoke == 0:
            time_since_ai = 999
        else: 
            time_since_ai = time.time() - last_ai_spoke
        
        result = classifier.should_reply(user_text, time_since_ai)
        
        status = "✅" if result == expected else "❌"
        print(f"{status} User: \"{user_text}\"")
        print(f"   ⏱ {time_since_ai:.1f}s since AI | Expected: {expected} | Got: {result}")
        print(f"   📝 {note}")
        
        if result: 
            print(f"   🤖 AI responds...")
            last_ai_spoke = time.time()
        print()
