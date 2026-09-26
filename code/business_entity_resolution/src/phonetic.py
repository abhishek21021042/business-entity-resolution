"""
Pure-Python implementation of the New York State Identification and Intelligence System (NYSIIS)
phonetic algorithm. Provides zero-dependency, platform-independent phonetic indexing.
"""

import re


def nysiis(name: str) -> str:
    """
    Computes NYSIIS phonetic code for an input word/string.
    """
    if not name:
        return ""

    s = re.sub(r"[^A-Z]", "", name.upper().strip())
    if not s:
        return ""

    # Step 1: Initial letter transformations
    if s.startswith("MAC"):
        s = "MCC" + s[3:]
    elif s.startswith("KN"):
        s = "NN" + s[2:]
    elif s.startswith("K"):
        s = "C" + s[1:]
    elif s.startswith(("PH", "PF")):
        s = "FF" + s[2:]
    elif s.startswith("SCH"):
        s = "SSS" + s[3:]

    # Step 2: Final letter transformations
    if s.endswith(("EE", "IE")):
        s = s[:-2] + "Y"
    elif s.endswith(("DT", "RT", "RD", "NT", "ND")):
        s = s[:-2] + "D"

    # Step 3: Key creation loop
    key = [s[0]]
    i = 1
    n = len(s)

    while i < n:
        char = s[i]
        nxt = s[i + 1] if i + 1 < n else ""

        # EV -> AF
        if char == "E" and nxt == "V":
            key.append("AF")
            i += 2
            continue

        # Vowels
        if char in "AEIOU":
            key.append("A")
            i += 1
            continue

        # Q -> G, Z -> S, M -> N
        if char == "Q":
            key.append("G")
        elif char == "Z":
            key.append("S")
        elif char == "M":
            key.append("N")
        elif char == "K":
            if nxt == "N":
                key.append("N")
                i += 1
            else:
                key.append("C")
        elif char == "S" and nxt == "C" and (i + 2 < n and s[i + 2] == "H"):
            key.append("S")
            i += 2
        elif char == "P" and nxt == "H":
            key.append("F")
            i += 1
        elif char == "H" and (s[i - 1] not in "AEIOU" or (nxt and nxt not in "AEIOU")):
            # H preceded or followed by non-vowel
            pass
        elif char == "W" and s[i - 1] in "AEIOU":
            pass
        else:
            key.append(char)

        i += 1

    res_str = "".join(key)

    # Step 4: Deduplicate consecutive identical characters
    dedup = [res_str[0]] if res_str else []
    for c in res_str[1:]:
        if c != dedup[-1]:
            dedup.append(c)

    res = "".join(dedup)

    # Step 5: Final vowel check
    if res.endswith("S") and len(res) > 1:
        res = res[:-1]
    if res.endswith("AY"):
        res = res[:-2] + "Y"
    if res.endswith("A") and len(res) > 1:
        res = res[:-1]

    return res


def phonetic_key(text: str) -> str:
    """Computes whitespace-joined NYSIIS keys for all words in text."""
    if not text:
        return ""
    words = re.findall(r"[A-Za-z]+", text)
    return " ".join(nysiis(w) for w in words if w)


if __name__ == "__main__":
    assert nysiis("Smith") == "SNAT"
    print("NYSIIS tests passed! Sample: Smith ->", nysiis("Smith"), "| Johnson ->", nysiis("Johnson"))
