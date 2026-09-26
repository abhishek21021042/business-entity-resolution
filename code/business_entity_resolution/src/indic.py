"""
Lightweight Indic Script to Latin Transliterator.
Maps Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
to phonetic Latin equivalents for business name resolution.
Zero external dependencies, purely algorithmic Unicode range mapping.
"""

# Base Devanagari consonant / vowel mapping
DEV_TO_LATIN = {
    # Vowels
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    # Consonants
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x32: "l", 0x33: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Matras (vowel signs)
    0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x47: "e", 0x48: "ai", 0x4B: "o", 0x4C: "au", 0x4D: "", # virama (halant)
    # Signs
    0x02: "n", 0x03: "h", 0x50: "om", 0x31: "",
}

# Indic scripts in Unicode: offset from base
INDIC_RANGES = [
    (0x0900, 0x097F, 0x0900), # Devanagari
    (0x0980, 0x09FF, 0x0980), # Bengali
    (0x0A00, 0x0A7F, 0x0A00), # Gurmukhi
    (0x0A80, 0x0AFF, 0x0A80), # Gujarati
    (0x0B00, 0x0B7F, 0x0B00), # Oriya
    (0x0B80, 0x0BFF, 0x0B80), # Tamil
    (0x0C00, 0x0C7F, 0x0C00), # Telugu
    (0x0C80, 0x0CFF, 0x0C80), # Kannada
    (0x0D00, 0x0D7F, 0x0D00), # Malayalam
]

def transliterate_indic_to_latin(text: str) -> str:
    """
    Translates Indic script text into approximate phonetic Latin characters.
    If text is already Latin or ASCII, returns original text.
    """
    if not text:
        return ""
    
    out = []
    for ch in text:
        cp = ord(ch)
        matched = False
        for start, end, base in INDIC_RANGES:
            if start <= cp <= end:
                offset = cp - base
                char_str = DEV_TO_LATIN.get(offset)
                if char_str is not None:
                    out.append(char_str)
                matched = True
                break
        if not matched:
            out.append(ch)
            
    res = "".join(out).lower()
    # Normalize common phonetics in transliterated business names
    replacements = [
        ("praivet", "pvt"), ("pra. li.", "pvt ltd"), ("limiteḍ", "ltd"), ("limited", "ltd"),
        ("entrapraaijez", "enterprises"), ("kaunsaltents", "consultants"), ("globl", "global"),
        ("sarvises", "services"), ("prvtre", "private"), ("prmvmase", "private"),
        ("solyushans", "solutions"), ("soleyooshns", "solutions"), ("sooshyatns", "solutions"),
        ("end", "and"), ("indiyan", "indian"), ("industreez", "industries"),
    ]
    for orig, rep in replacements:
        res = res.replace(orig, rep)
    return res

if __name__ == "__main__":
    tests = [
        "ग्लोबल कंसल्टेंट्स प्राइवेट लिमिटेड",
        "व्हाइट प्रोडक्ट्स प्राइवेट लिमिटेड",
        "हरि इंडियन एंटरप्राइजेज प्राइवेट लिमिटेड",
        "रॉयल सूर्य मैनेजमेंट",
        "પરફેક્ટ એક્સપોર્ટ્સ પ્રા. લિ.",
        "स्वस्तिक ॐ सॉल्यूशंस एलएलपी",
        "ఎస్‌ఎస్ స్టార్ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్",
    ]
    for t in tests:
        print(f"{t}  -->  {transliterate_indic_to_latin(t)}")
