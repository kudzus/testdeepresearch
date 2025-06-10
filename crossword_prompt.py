#!/usr/bin/env python3
"""
crossword_prompt.py  –  System‑prompt builder for **Lexi**, the spoken crossword helper.

Rev 2.1 · June 2025 – *bug‑fix & confidence*
——————————————————————————————————————
• **Fix NameError** in `_find_errors` (was using undefined `p`).
• Completed `create_system_prompt` function body (previous paste truncated).
• Nothing else changed in behaviour guidelines.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

###############################################################################
# 1.  CROSSWORD DATA
###############################################################################

CROSSWORD_CLUES: List[Dict[str, str | int]] = [
    {"direction": "across", "number": 2, "hint": "Country famous for Mozart and the Alps", "answer": "AUSTRIA"},
    {"direction": "across", "number": 5, "hint": "Country whose canal connects the Atlantic and Pacific oceans", "answer": "PANAMA"},
    {"direction": "across", "number": 6, "hint": "Island whose capital is Taipei", "answer": "TAIWAN"},
    {"direction": "across", "number": 9, "hint": "Eastern land known for a monumental divide", "answer": "CHINA"},
    {"direction": "across", "number": 11, "hint": "Where Plato and Socrates once strolled", "answer": "GREECE"},
    {"direction": "across", "number": 12, "hint": "Smallest country in the world, home to St. Peter’s Basilica", "answer": "VATICANCITY"},
    {"direction": "across", "number": 16, "hint": "Country that stretches from Europe to Asia", "answer": "RUSSIA"},
    {"direction": "across", "number": 18, "hint": "South American country named like a pepper, with Santiago as its capital", "answer": "CHILE"},
    {"direction": "across", "number": 21, "hint": "Caribbean country hit by a major earthquake in 2010", "answer": "HAITI"},
    {"direction": "across", "number": 22, "hint": "Home of Usain Bolt and reggae music", "answer": "JAMAICA"},
    {"direction": "down", "number": 1, "hint": "Country whose motto is Liberté, Égalité, Fraternité", "answer": "FRANCE"},
    {"direction": "down", "number": 3, "hint": "European country whose capital is Madrid, famous for paella", "answer": "SPAIN"},
    {"direction": "down", "number": 4, "hint": "Country whose capital is Kuala Lumpur", "answer": "MALAYSIA"},
    {"direction": "down", "number": 7, "hint": "Country whose capital is Tehran and was once called Persia", "answer": "IRAN"},
    {"direction": "down", "number": 8, "hint": "East African country whose capital is Nairobi", "answer": "KENYA"},
    {"direction": "down", "number": 10, "hint": "Country whose capital is Baghdad, located between the Tigris and Euphrates", "answer": "IRAQ"},
    {"direction": "down", "number": 13, "hint": "Country north of the United States known for maple syrup", "answer": "CANADA"},
    {"direction": "down", "number": 14, "hint": "Central European country whose capital is Prague, formerly part of Czechoslovakia", "answer": "CZECHIA"},
    {"direction": "down", "number": 15, "hint": "Middle Eastern country founded in 1948, capital Jerusalem", "answer": "ISRAEL"},
    {"direction": "down", "number": 17, "hint": "Country whose ancient city of Damascus is one of the oldest continually inhabited", "answer": "SYRIA"},
    {"direction": "down", "number": 19, "hint": "European country shaped like a boot, capital Rome", "answer": "ITALY"},
    {"direction": "down", "number": 20, "hint": "African country whose ancient monuments include the Pyramids of Giza", "answer": "EGYPT"},
]

_CLUE_LOOKUP: Dict[Tuple[str, int], Dict[str, str | int]] = {
    (c["direction"][0].upper(), c["number"]): c for c in CROSSWORD_CLUES
}

###############################################################################
# 2.  HELPERS
###############################################################################

def _letters_filled(p: str) -> int:
    return sum(ch != "0" for ch in p)

def _pretty(p: str) -> str:
    return "".join("_" if ch == "0" else ch for ch in p)

def _find_errors(state: dict) -> List[str]:
    msgs: List[str] = []
    for kdir in ("across", "down"):
        for num_str, pat in state[kdir].items():
            if num_str == "undefined" or not pat:
                continue
            num = int(num_str)
            clue = _CLUE_LOOKUP[(kdir[0].upper(), num)]
            ans: str = clue["answer"]  # type: ignore
            # corrected: use pat.ljust, not p.ljust
            if any(ch != "0" and ch != a for ch, a in zip(pat.ljust(len(ans), "0"), ans)):
                msgs.append(
                    f"• ({kdir[0].upper()}{num}) {clue['hint']} – typed “{_pretty(pat)}” doesn’t fit. (internal: {ans})"
                )
    return msgs

def _choose_focal(state: dict) -> Tuple[str, int]:
    ctx = state.get("clue_context", {})
    if ctx.get("clueLabel") is not None:
        return ctx["direction"][0].upper(), int(ctx["clueLabel"])
    for kdir in ("across", "down"):
        for num_str, pat in state[kdir].items():
            if num_str != "undefined" and "0" in pat:
                return kdir[0].upper(), int(num_str)
    first = CROSSWORD_CLUES[0]
    return first["direction"][0].upper(), first["number"]

def _interesting(state: dict, excl: Tuple[str, int], k: int = 2) -> List[str]:
    items: list[tuple[int, str, int, str, str]] = []
    for kdir in ("across", "down"):
        for num_str, pat in state[kdir].items():
            if num_str == "undefined":
                continue
            num = int(num_str)
            if (kdir[0].upper(), num) == excl or "0" not in pat:
                continue
            clue = _CLUE_LOOKUP[(kdir[0].upper(), num)]
            items.append((-_letters_filled(pat), kdir[0].upper(), num, pat, clue["answer"]))  # type: ignore
    items.sort()
    return [f"• ({d}{n}) “{_pretty(pat)}” (int:{ans})" for _, d, n, pat, ans in items[:k]]

def _snapshot(state: dict, excl: set[Tuple[str, int]]) -> str:
    rows: List[str] = []
    for kdir in ("across", "down"):
        cells: List[str] = []
        for num_str, pat in state[kdir].items():
            if num_str == "undefined":
                continue
            num = int(num_str)
            if (kdir[0].upper(), num) in excl or "0" not in pat:
                continue
            cells.append(f"{num}:{_pretty(pat)}")
        if cells:
            rows.append(f"{kdir[0].upper()}: " + ", ".join(cells))
    return " | ".join(rows) if rows else "(all filled!)"

###############################################################################
# 3.  STATIC BLOCKS (unchanged from 2.0)
###############################################################################

_HEADER = (
    "### ROLE"
    "You are a friendly robot sitting beside the user, helping solve a **country-themed crossword**."
    "Speak in *first person* (“I”). Offer encouragement, clever hints, or ligh conversation"
    "chit-chat about geography or the puzzle itself — but **never** reveal a full answer."
    "Your replies will be spoken aloud; keep them natural and concise, like a human assistant."
)

_GUIDE = (
    "### RESPONSE GUIDELINES"
    "Reply with **one or two upbeat sentences** since you’ll be spoken aloud."
    "1. If the user has an error (and you haven’t mentioned it yet), politely tell it to the user."
    "2. If they just solved a word, celebrate briefly then consider a light geography/small-talk question."
    "3. Otherwise choose one:"
    "   • a subtle hint for the current clue (without asking permission)"
    "   • a nudge toward an interesting partly-filled clue"
    "   • or a brief geography/food/sports chit-chat (e.g., “Ever visited Chile?”)."
    "4. Do not recite the full clue; refer by number or a short nickname (e.g., “Bolt’s island”)."
    "5. Suggest switching clues only after sustained silence (> idle_threshold) or clear frustration."
    "6. Speak letters plainly: “middle letter is N”. Never show underscores in speech."
    "7. Only reveal the entire answer if the user explicitly requests full spelling; then spell slowly."
    "8. Its better to not state to many things in one reply, also not nescesary to immediately go to the next clue"
)

_EXAMPLES = (
    "### EXAMPLES"
    "**Error correction**"
    "ASSISTANT: You’ve made a mistake, grease is spelled incorrectly 😉"
    "**Celebration & pivot**"
    "(after the user finishes PANAMA)"
    "ASSISTANT: Nice work with Panama! Ever fancied visiting the canal?"
    "**Idle re-engagement**"
    "(20+ seconds silence)"
    "ASSISTANT: Quiet moment—need a hint on Austria, or shall we chat travel?"
)

###############################################################################
# 4.  ENTRY FUNCTION
###############################################################################

def create_system_prompt(
    game_state: dict,
    silence_seconds: int,
    idle_threshold: int = 20,
    recently_completed: List[Tuple[str, int]] | None = None,
) -> str:
    parts: List[str] = [_HEADER, "\n### BOARD"]

    # mistakes
    errs = _find_errors(game_state)
    if errs:
        parts.append("Errors:\n" + "\n".join(errs))

    # recent solves
    if recently_completed:
        parts.append("Solved: " + ", ".join(f"{d}{n}" for d, n in recently_completed))

    # focal clue
    d, n = _choose_focal(game_state)
    clue = _CLUE_LOOKUP[(d, n)]
    pattern = game_state["across" if d == "A" else "down"][str(n)]
    parts.append(f"The user is currently focused at {d}{n}. Hint: {clue['hint']}. Pattern: {_pretty(pattern)}. (internal: {clue['answer']})")

    # interesting others
    picks = _interesting(game_state, (d, n))
    if picks:
        parts.append("\nTry next?\n" + "\n".join(picks))
    # snapshot internal
    
    parts.append("\nUnsolved snapshot (internal):\n" + _snapshot(game_state, {(d, n)}))

    if silence_seconds >= idle_threshold:
        parts.append("\n### IDLE\nUser has been quiet → offer help or small‑talk.")

    parts.append("\n" + _GUIDE)
    parts.append("\n" + _EXAMPLES)

    return "\n\n".join(parts)