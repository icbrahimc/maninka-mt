"""N'Ko script -> Latin Maninka transliteration, plus a normalizer so that
N'Ko-derived text, Latin Maninka (Guinean/Bambara-style ɛ ɔ ɲ ŋ) and loose song
transcriptions (e/o/ny spellings) all end up in ONE canonical form.

Canonical "plain" form (default): lowercase ASCII-ish, e/o for ɛ/ɔ, ny for ɲ,
ng for ŋ, no tone marks, long vowels doubled (aa, ee...), nasal vowels + n.
This loses the ɛ/e and ɔ/o contrast, deliberately: song transcripts don't keep
it, and a consistent spelling matters more than a precise one.

Use style="ipa" if you later want to keep ɛ ɔ ɲ ŋ (closer to NLLB's Bambara).
"""
from __future__ import annotations

import re
import unicodedata

# --- base letters (U+07C0..U+07EA) ------------------------------------------
_DIGITS = {chr(0x07C0 + i): str(i) for i in range(10)}

_VOWELS = {
    "ߊ": "a",  # ߊ
    "ߋ": "e",  # ߋ
    "ߌ": "i",  # ߌ
    "ߍ": "ɛ",  # ߍ
    "ߎ": "u",  # ߎ
    "ߏ": "o",  # ߏ
    "ߐ": "ɔ",  # ߐ
}

_CONSONANTS = {
    "ߒ": "n",   # ߒ syllabic n
    "ߓ": "b",   # ߓ
    "ߔ": "p",   # ߔ
    "ߕ": "t",   # ߕ
    "ߖ": "j",   # ߖ
    "ߗ": "c",   # ߗ
    "ߘ": "d",   # ߘ
    "ߙ": "r",   # ߙ
    "ߚ": "rr",  # ߚ
    "ߛ": "s",   # ߛ
    "ߜ": "gb",  # ߜ
    "ߝ": "f",   # ߝ
    "ߞ": "k",   # ߞ
    "ߟ": "l",   # ߟ
    "ߠ": "n",   # ߠ na woloso (l -> n after nasals)
    "ߡ": "m",   # ߡ
    "ߢ": "ɲ",   # ߢ
    "ߣ": "n",   # ߣ
    "ߤ": "h",   # ߤ
    "ߥ": "w",   # ߥ
    "ߦ": "y",   # ߦ
    "ߧ": "ɲ",   # ߧ nya woloso
    "ߨ": "j",   # archaic jona ja
    "ߩ": "c",   # archaic jona cha
    "ߪ": "r",   # archaic jona ra
}

# Foreign-sound modifiers (dot above). Best-effort; uncommon in this corpus.
_FOREIGN = {("ߜ", "g"), ("ߖ", "z"), ("ߝ", "v")}
_FOREIGN_MAP = dict(_FOREIGN)
_FOREIGN_MARKS = {"߭", "߳"}

_SHORT_TONES = {"߫", "߬", "߭"}            # dropped
_LONG_TONES = {"߮", "߯", "߰", "߱"}   # -> double vowel
_NASAL = "߲"                                        # -> vowel + n
_DAGBASINNA = "ߑ"                                   # ߑ "no vowel" mark
_PUNCT = {
    "߸": ",",   # ߸
    "߹": "!",   # ߹
    "ߴ": "'",   # ߴ high-tone apostrophe (elision: ߞߵߊ -> k'a)
    "ߵ": "'",   # ߵ low-tone apostrophe
    "ߺ": "",    # ߺ lajanyalan (elongation, typographic)
    "،": ",",   # Arabic comma, sometimes used in N'Ko text
    "؟": "?",   # Arabic question mark
    "؛": ";",
}


_MARKS = _SHORT_TONES | _LONG_TONES | {_NASAL} | _FOREIGN_MARKS


def _restore_elided_vowels(text: str) -> str:
    """N'Ko omits a vowel when consecutive syllables share it: ߦߟߍ is written
    y-l-ɛ but read yɛlɛ; ߝߌߛߙߌ is f-i-s-r-i -> fisiri. Re-insert the vowel
    whenever a consonant letter is directly followed by another consonant
    letter, copying the next vowel in the word. ߒ (syllabic n) and consonants
    marked with ߑ (dagbasinna, an explicit 'no vowel') are left alone."""
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        c = chars[i]
        if c in _CONSONANTS and c != "ߒ":
            j = i + 1
            while j < len(chars) and chars[j] in _FOREIGN_MARKS:
                j += 1
            nxt = chars[j] if j < len(chars) else ""
            if nxt in _CONSONANTS and nxt != "ߒ":
                k = j
                while k < len(chars) and chars[k] not in _VOWELS and (
                    chars[k] in _CONSONANTS or chars[k] in _MARKS
                    or chars[k] in ("\u07F4", "\u07F5")
                ):
                    k += 1
                if k < len(chars) and chars[k] in _VOWELS:
                    chars.insert(j, chars[k])
                    i = j
        i += 1
    return "".join(chars)


def nko_to_latin(text: str, style: str = "plain") -> str:
    """Transliterate N'Ko text. Non-N'Ko characters pass through unchanged."""
    text = _restore_elided_vowels(text)
    out: list[str] = []
    last_vowel: str | None = None   # vowel of the current syllable
    nasal_pending = False           # nasal mark already appended "n"

    for ch in text:
        if ch in _VOWELS:
            v = _VOWELS[ch]
            out.append(v)
            last_vowel, nasal_pending = v, False
        elif ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            last_vowel, nasal_pending = None, False
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
            last_vowel = None
        elif ch in _FOREIGN_MARKS and out and any(
            out[-1] == _CONSONANTS[k] for k in _FOREIGN_MAP
        ):
            for k, repl in _FOREIGN_MAP.items():
                if out[-1] == _CONSONANTS[k]:
                    out[-1] = repl
                    break
        elif ch in _LONG_TONES:
            if last_vowel:
                if nasal_pending:          # "an" + long -> "aan"
                    out.insert(len(out) - 1, last_vowel)
                else:
                    out.append(last_vowel)
        elif ch in _SHORT_TONES or ch == _DAGBASINNA:
            pass
        elif ch == _NASAL:
            if last_vowel and not nasal_pending:
                out.append("n")
                nasal_pending = True
        elif ch in _PUNCT:
            out.append(_PUNCT[ch])
            last_vowel = None
        else:
            out.append(ch)
            if not ch.isalpha():
                last_vowel = None

    s = "".join(out)
    return normalize(s, style=style)


# --- normalizer for ANY Maninka Latin text -----------------------------------
_PLAIN_MAP = str.maketrans({
    "ɛ": "e", "ɔ": "o", "è": "e", "ò": "o", "é": "e", "ó": "o",
    "à": "a", "á": "a", "ì": "i", "í": "i", "ù": "u", "ú": "u",
    "’": "'", "‘": "'", "ʼ": "'",
})


def normalize(text: str, style: str = "plain") -> str:
    """Canonicalize Maninka Latin text (from N'Ko, a dictionary, or lyrics)."""
    s = unicodedata.normalize("NFC", text).lower()
    s = s.replace("’", "'").replace("‘", "'")
    if style == "plain":
        s = s.replace("ɲ", "ny").replace("ŋ", "ng")
        s = s.translate(_PLAIN_MAP)
        # strip any remaining combining tone marks (grave/acute/caron/etc.)
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if not unicodedata.combining(c))
    elif style == "ipa":
        s = s.replace("ny", "ɲ").replace("ng", "ŋ")
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if not unicodedata.combining(c))
        s = unicodedata.normalize("NFC", s)
    else:
        raise ValueError(f"unknown style {style!r}")
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_nko(text: str) -> bool:
    return any("߀" <= c <= "߿" for c in text)


def to_canonical(text: str, style: str = "plain") -> str:
    """One entry point for any Maninka input, N'Ko or Latin."""
    return nko_to_latin(text, style) if is_nko(text) else normalize(text, style)


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print(to_canonical(line.rstrip("\n")))
