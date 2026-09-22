# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

import string


# ---------------------------------------------------------------------
# Emoji helpers (module level so preprocess can call them directly)
# ---------------------------------------------------------------------

# Text emoticons to keep intact (matched before lowercasing, so ":D" works).
EMOTICONS = [":-)", ":)", ":D", ":-(", ":(", ":'(", ":/", ";)", "<3"]

# Codepoint ranges that count as pictographic emoji.
EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),  # 😂 💀 🥲 and friends
    (0x2600, 0x27BF),    # ☀ ❤ ✨ dingbats
]


def _is_emoji(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES)


# ---------------------------------------------------------------------
# Scoring tables
# ---------------------------------------------------------------------

# Emojis and slang carry a STRONGER signal than an ordinary word, so they
# get weights of +/-2 while a plain word from the lists is worth +/-1.
# (Emoticon keys are lowercase because preprocess lowercases them.)
STRONG_SIGNALS: Dict[str, int] = {
    # emoticons
    ":)": 2, ":-)": 2, ":d": 2, "<3": 2, ";)": 1,
    ":(": -2, ":-(": -2, ":'(": -2, ":/": -1,
    # unicode emoji
    "😂": 2, "🥰": 2, "❤": 2, "✨": 1, "🔥": 2,
    "😭": -2, "😡": -2, "🥲": -1, "💀": -1,
    # slang
    "lol": 1, "lmao": 1, "slay": 2, "goated": 2, "baddie": 2, "vibes": 1,
    "meh": -1, "ugh": -1, "cringe": -2, "sucks": -2, "trash": -2, "bombed": -2,
}

# A negation word flips the sign of the next couple of scored tokens,
# so "not happy" reads negative and "not bad" reads positive.
NEGATION_WORDS = {
    "not", "no", "never", "none", "nothing", "cant", "cannot", "dont",
    "wont", "aint", "isnt", "wasnt", "hardly", "barely",
}

# How many tokens after a negation word stay flipped.
NEGATION_WINDOW = 2

# An intensifier multiplies the next scored token instead of adding to the
# score itself, so "so bad" hits harder than a plain "bad" and "so happy"
# harder than "happy".
INTENSIFIERS: Dict[str, int] = {
    "so": 2, "very": 2, "really": 2, "super": 2, "mad": 2,
    "extremely": 3, "absolutely": 3, "totally": 2,
}

# A "mixed" message has to argue with itself: both sides need at least this
# many points before the label is even considered...
MIXED_MIN_POINTS = 2

# ...and the two sides have to end up this close for neither to be the winner.
MIXED_TOLERANCE = 1


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Emojis are pulled out before punctuation is stripped, otherwise
        ":)" and ":-(" would be deleted entirely.
        """

        text = text.strip()
        emoji_tokens: List[str] = []

        # 1. Pull out text emoticons FIRST, while the punctuation is still there.
        #    Longest first so ":-)" isn't chewed up by a shorter pattern.
        for face in sorted(EMOTICONS, key=len, reverse=True):
            while face in text:
                emoji_tokens.append(face.lower())
                text = text.replace(face, " ", 1)

        text = text.lower()

        # 2. Split unicode emoji off into their own tokens.
        kept = []
        for ch in text:
            if _is_emoji(ch):
                emoji_tokens.append(ch)
            else:
                kept.append(ch)
        text = "".join(kept)

        # 3. Now it's safe to drop punctuation and split on whitespace.
        cleaned = text.translate(str.maketrans("", "", string.punctuation))

        return cleaned.split() + emoji_tokens


    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_breakdown(self, text: str) -> Tuple[int, int]:
        """
        Tally the positive and negative points in the text separately.

        Returns (positive_points, negative_points), both non-negative. The
        plain score is positive_points - negative_points, but keeping the two
        sides apart is what lets predict_label tell a genuinely two sided
        "mixed" message ("+3 and -3") from a flat one ("nothing scored").

        Points are counted AFTER the boost and the negation flip, so a token
        lands on whichever side it actually ends up on: in "not bad" the
        "bad" counts as a positive point, not a negative one.

        Modeling improvements implemented here:
          - Simple negation: a word from NEGATION_WORDS flips the sign of the
            next NEGATION_WINDOW scored tokens, so "not happy" is negative
            and "not bad" is positive.
          - Strong signals: emojis and slang from STRONG_SIGNALS are worth
            +/-2 instead of the +/-1 an ordinary word gets.
          - Intensifiers: a word from INTENSIFIERS ("so", "very", ...) scores
            nothing on its own but multiplies the next token that does score,
            so "so bad" is -2 where a plain "bad" is -1. The boost waits until
            it finds a scoring word, so filler in between ("so incredibly
            bad") does not waste it, and stacked intensifiers multiply
            ("so very bad" is -4).
          - Repeats count: every occurrence of a word adds to the score,
            not just the first one.
        """
        tokens = self.preprocess(text)

        positive_points = 0
        negative_points = 0
        flip_remaining = 0  # how many upcoming tokens the negation still affects
        boost = 1  # multiplier waiting to be spent on the next scoring token

        for token in tokens:
            # A negation word scores nothing itself; it arms the flip.
            if token in NEGATION_WORDS:
                flip_remaining = NEGATION_WINDOW
                continue

            # Neither does an intensifier; it just makes the next hit bigger.
            # It stays out of the negation window too, so "not so bad" still
            # flips onto "bad" and comes out positive.
            if token in INTENSIFIERS:
                boost *= INTENSIFIERS[token]
                continue

            # Emojis and slang win over the plain word lists because they
            # are the stronger signal.
            if token in STRONG_SIGNALS:
                value = STRONG_SIGNALS[token]
            elif token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                value = 0

            if value != 0:
                value *= boost
                boost = 1  # the boost is spent on one word only

            if flip_remaining > 0:
                value = -value
                flip_remaining -= 1

            # Each token adds to one pile or the other, never both.
            if value > 0:
                positive_points += value
            else:
                negative_points += -value

        return positive_points, negative_points

    def score_text(self, text: str) -> int:
        """
        Compute a single numeric "mood score": positives minus negatives.
        """
        positive_points, negative_points = self.score_breakdown(text)
        return positive_points - negative_points

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the scored text into a mood label.

        "mixed" is decided from the two piles of points, not from the net
        score, because a net of 0 can mean two very different things: a text
        that said nothing either way (neutral), or a text that said a lot in
        both directions at once (mixed). A text is "mixed" when both sides
        carry real weight and neither one clearly wins:
          - at least MIXED_MIN_POINTS on the positive side, AND
          - at least MIXED_MIN_POINTS on the negative side, AND
          - the two sides within MIXED_TOLERANCE points of each other

        Anything else falls back to the net score:
          - score > 0   -> "positive"
          - score < 0   -> "negative"
          - score == 0  -> "neutral"

        These labels match the ones used in TRUE_LABELS in dataset.py.
        """
        positive_points, negative_points = self.score_breakdown(text)

        both_sides_show_up = (
            positive_points >= MIXED_MIN_POINTS
            and negative_points >= MIXED_MIN_POINTS
        )
        sides_are_close = abs(positive_points - negative_points) <= MIXED_TOLERANCE

        if both_sides_show_up and sides_are_close:
            return "mixed"

        score = positive_points - negative_points

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
