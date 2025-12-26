import logging
from typing import Optional, Set, Tuple, Dict, Union # Added for type hinting

logger = logging.getLogger(__name__)
from colors import Colors # Assuming this is needed externally

class TextContext:
    """
    Extracts meaningful text segments (contexts) from a given string.

    This class identifies substrings that end with predefined split tokens,
    adhere to specified length constraints, and contain a minimum number
    of alphanumeric characters.
    """
    def __init__(self, split_tokens: Optional[Set[str]] = None) -> None:
        """
        Initializes the TextContext processor.

        Sets up the characters used to determine valid context boundaries.

        Args:
            split_tokens: An optional set of strings. Each string is treated as a
                          potential end-of-context marker. If None, a default set
                          of punctuation and whitespace characters is used.
        """
        if split_tokens is None:
            # Using a more explicit variable name internally for clarity
            default_splits: Set[str] = {".", "!", "?", ",", ";", ":", "\n", "-", "。", "、", "…"}  # Add ellipsis
            self.split_tokens: Set[str] = default_splits
        else:
            self.split_tokens: Set[str] = set(split_tokens)

    def get_context(self, txt: str, min_len: int = 6, max_len: int = 120, min_alnum_count: int = 10) -> Tuple[Optional[str], Optional[str]]:
        """
        Finds the shortest valid context at the beginning of the input text.

        Scans the text `txt` from the beginning up to `max_len` characters. It looks
        for the first occurrence of a character from `self.split_tokens`. If found,
        it checks if the substring ending at that token meets the `min_len` (overall
        length) and `min_alnum_count` (alphanumeric character count) criteria.

        Args:
            txt: The input string from which to extract the context.
            min_len: The minimum allowable overall length for the extracted context substring.
            max_len: The maximum allowable overall length for the extracted context substring.
                     The search stops after examining this many characters.
            min_alnum_count: The minimum number of alphanumeric characters required within
                             the extracted context substring.

        Returns:
            A tuple containing:
            - The extracted context string if found, otherwise None.
            - The remaining part of the input string after the context, otherwise None.
            Returns (None, None) if no suitable context is found within the constraints.
        """
        alnum_count = 0

        # Check if text ends with a potential incomplete time pattern
        # Patterns like "20:", "09:", "двадцать:", "тринадцать:" etc.
        incomplete_time_patterns = [
            r'\d{1,2}: $',  # "20:" or "9:"
            r'(? : ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|семнадцать|восемнадцать|девятнадцать|двадцать|двадцать\s+(? : один|два|три|четыре)):$',
        ]

        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1

            # Check if the current character is a potential context end
            if char in self.split_tokens:
                # Check if length and alphanumeric count criteria are met
                if i >= min_len and alnum_count >= min_alnum_count:
                    context_str = txt[:i]
                    remaining_str = txt[i:]
                    logger.info(f"🧠 {Colors.MAGENTA}Context found after char no: {i}, context: {context_str}")
                    return context_str, remaining_str

        # No suitable context found within the max_len limit
        return None, None