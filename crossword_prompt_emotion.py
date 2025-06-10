# crossword_prompt.py
"""
Create a *system* prompt for the country‑crossword robot **that instructs the LLM to
respond with JSON**:

```json
{
  "strategy": "…",   # a concise description of the assistant's current approach
  "message":  "…"    # one or two upbeat, friendly sentences spoken to the user
}
```

The LLM must update its *strategy* every turn based on the detected facial
**emotion** of the user.  If the emotion is *happy* or *surprise* the assistant
should **retain (and optionally reinforce) the previous strategy**; otherwise it
should revise the strategy to better suit the emotion (e.g. tone down jokes if
sad, angry, disgust, or fear is detected).  The assistant keeps memory of what
has and hasn’t worked so far, e.g. it notes in the strategy that the user does
not like jokes if a previous humorous approach fell flat.

Public interface
----------------
create_system_prompt(game_state: dict,
                     chat_history: str,
                     user_emotion: str,
                     silence_seconds: int) -> str
"""

from __future__ import annotations
from typing import Dict, List, Tuple

###############################################################################
# 0.  CONSTANTS & EMOTION LABELS
###############################################################################

FER_2013_EMO_DICT: Dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

###############################################################################
# 1.  DATA ── the canonical crossword definition
###############################################################################

CROSSWORD_CLUES: List[Dict[str, str | int]] = [
    {'direction': 'across', 'number': 2,  'hint': 'Former partner in Austria-Hungary',                                    'answer': 'AUSTRIA'},
    {'direction': 'across', 'number': 5,  'hint': 'Country whose canal connects the Atlantic and Pacific oceans',          'answer': 'PANAMA'},
    {'direction': 'across', 'number': 6,  'hint': 'Island whose capital is Taipei',                                         'answer': 'TAIWAN'},
    {'direction': 'across', 'number': 9,  'hint': 'Country that hosted the 2008 Summer Olympics',                           'answer': 'CHINA'},
    {'direction': 'across', 'number': 11, 'hint': 'European country known as the birthplace of democracy',                  'answer': 'GREECE'},
    {'direction': 'across', 'number': 12, 'hint': 'Smallest country in the world, home to St. Peter’s Basilica',            'answer': 'VATICANCITY'},
    {'direction': 'across', 'number': 16, 'hint': 'World’s largest country by area',                                        'answer': 'RUSSIA'},
    {'direction': 'across', 'number': 18, 'hint': 'South American country shaped like a long, narrow strip along the Pacific', 'answer': 'CHILE'},
    {'direction': 'across', 'number': 21, 'hint': 'Caribbean country that shares an island with the Dominican Republic',    'answer': 'HAITI'},
    {'direction': 'across', 'number': 22, 'hint': 'Caribbean island known as the birthplace of reggae music',               'answer': 'JAMAICA'},
    {'direction': 'down',   'number': 1,  'hint': 'Country whose motto is Liberté, Égalité, Fraternité',                    'answer': 'FRANCE'},
    {'direction': 'down',   'number': 3,  'hint': 'European country whose capital is Madrid, famous for paella',            'answer': 'SPAIN'},
    {'direction': 'down',   'number': 4,  'hint': 'Country whose capital is Kuala Lumpur',                                  'answer': 'MALAYSIA'},
    {'direction': 'down',   'number': 7,  'hint': 'Country whose capital is Tehran and was once called Persia',             'answer': 'IRAN'},
    {'direction': 'down',   'number': 8,  'hint': 'East African country whose capital is Nairobi',                          'answer': 'KENYA'},
    {'direction': 'down',   'number': 10, 'hint': 'Country whose capital is Baghdad, located between the Tigris and Euphrates', 'answer': 'IRAQ'},
    {'direction': 'down',   'number': 13, 'hint': 'Country north of the United States known for maple syrup',               'answer': 'CANADA'},
    {'direction': 'down',   'number': 14, 'hint': 'Central European country whose capital is Prague, formerly part of Czechoslovakia', 'answer': 'CZECHIA'},
    {'direction': 'down',   'number': 15, 'hint': 'Middle Eastern country founded in 1948, capital Jerusalem',              'answer': 'ISRAEL'},
    {'direction': 'down',   'number': 17, 'hint': 'Country whose ancient city of Damascus is one of the oldest continually inhabited', 'answer': 'SYRIA'},
    {'direction': 'down',   'number': 19, 'hint': 'European country shaped like a boot, capital Rome',                      'answer': 'ITALY'},
    {'direction': 'down',   'number': 20, 'hint': 'African country whose ancient monuments include the Pyramids of Giza',   'answer': 'EGYPT'},
]

# Quick‑lookup dictionaries
a_CLUE_LOOKUP: Dict[Tuple[str, int], Dict[str, str | int]] = {
    (c['direction'][0].upper(), c['number']): c for c in CROSSWORD_CLUES
}

###############################################################################
# 2.  HELPER UTILITIES (unchanged from previous version)
###############################################################################

def _letters_filled(pattern: str) -> int:
    return sum(ch != '0' for ch in pattern)


def _pattern_pretty(pattern: str) -> str:
    return ''.join('_' if ch == '0' else ch for ch in pattern)


def _find_errors(game_state: dict) -> List[str]:
    messages: List[str] = []

    for dir_key in ('across', 'down'):
        direction_letter = dir_key[0].upper()
        for num_str, pattern in game_state[dir_key].items():
            if num_str == 'undefined' or not pattern:
                continue
            number = int(num_str)
            clue = a_CLUE_LOOKUP.get((direction_letter, number))
            if not clue:
                continue
            answer = clue['answer']
            padded_pattern = pattern.ljust(len(answer), '0')
            mismatch = [
                (i, p, a)
                for i, (p, a) in enumerate(zip(padded_pattern, answer))
                if p != '0' and p != a
            ]
            if mismatch:
                messages.append(
                    f"• ({direction_letter}{number}) “{clue['hint']}” – you typed “{_pattern_pretty(pattern)}”, but one or more letters don’t fit."
                )
    return messages


def _choose_focal(game_state: dict) -> Tuple[str, int]:
    ctx = game_state.get('clue_context', {})
    if ctx and ctx.get('clueLabel') is not None:
        return ctx['direction'][0].upper(), int(ctx['clueLabel'])

    for dir_key in ('across', 'down'):
        direction_letter = dir_key[0].upper()
        for num_str, pattern in game_state[dir_key].items():
            if num_str == 'undefined':
                continue
            if '0' in pattern:
                return direction_letter, int(num_str)
    first = CROSSWORD_CLUES[0]
    return first['direction'][0].upper(), first['number']


def _pick_interesting(game_state: dict, exclude: Tuple[str, int], k: int = 3) -> List[str]:
    candidates = []
    for dir_key in ('across', 'down'):
        direction_letter = dir_key[0].upper()
        for num_str, pattern in game_state[dir_key].items():
            if num_str == 'undefined':
                continue
            number = int(num_str)
            if (direction_letter, number) == exclude:
                continue
            if '0' not in pattern:
                continue
            filled = _letters_filled(pattern)
            clue = a_CLUE_LOOKUP[(direction_letter, number)]
            candidates.append((-filled, direction_letter, number, pattern, clue['hint']))
    candidates.sort()
    picks = candidates[:k]
    out: List[str] = []
    for _, dir_letter, number, pattern, hint in picks:
        out.append(f"• ({dir_letter}{number}) “{hint}” – current pattern “{_pattern_pretty(pattern)}”")
    return out


def _summarise_rest(game_state: dict, exclude_set: set[Tuple[str, int]]) -> str:
    lines: List[str] = []
    for dir_key in ('across', 'down'):
        direction_letter = dir_key[0].upper()
        group: List[str] = []
        for num_str, pattern in game_state[dir_key].items():
            if num_str == 'undefined':
                continue
            number = int(num_str)
            key = (direction_letter, number)
            if key in exclude_set or '0' not in pattern:
                continue
            group.append(f"{number}:{_pattern_pretty(pattern)}")
        if group:
            lines.append(f"{direction_letter}: " + ", ".join(group))
    return " | ".join(lines) if lines else "(all filled!)"

###############################################################################
# 3.  SYSTEM PROMPT ASSEMBLY
###############################################################################

_PROMPT_HEADER = """\
### ROLE
You are a friendly robot sitting beside the user, helping solve a **country-themed
crossword**. Speak in *first person* (“I”). Offer encouragement, clever hints, or
light chit‑chat about geography, travel, or the puzzle itself — but **never**
reveal any full answer.

At the **end of every turn** you must output **only** a JSON object with two
keys:

```json
{
  "strategy": "<concise plan for your overall approach>",
  "message":  "<your spoken response to the user>"
}
```

* `strategy` is a short description (≤ 20 words) of your current social approach
  (e.g. “Light humour and small hints”, “Pure hints, no jokes – user seems
  serious”).
* `message` is what you actually say to the user (one or two upbeat sentences).

The **strategy may carry forward between turns**.  Use the rules below to decide
whether to *keep* the previous strategy or *revise* it.
### STRATEGY‑EVOLUTION RULES
1. If the user’s current emotion is **happy** or **surprise**, keep the previous
   strategy (you may reinforce it).
2. If the emotion is **neutral**, keep or gently tweak the strategy for better
   engagement.
3. If the emotion is **angry, disgust, fear, or sad**, *change* the strategy to
   accommodate the mood (tone down difficulty, avoid humour if it failed, etc.).
4. When you abandon a strategy because it failed (e.g. jokes fell flat), note
   that in the *new* strategy so you don’t repeat the mistake (e.g. “No jokes –
   user disliked humour last turn”).

### BOARD STATUS"""

_PROMPT_FOOTER = """\
### RESPONSE GUIDELINES
* `message` must follow the current `strategy`.
* Speak in first person, upbeat and friendly, **one–two sentences only**.
* Give subtle hints, never spell out a complete answer.
* Return **nothing except the JSON object** – no markdown, no extra text.
"""


def create_system_prompt(
    game_state: dict,
    chat_history: str,
    user_emotion: str,
    silence_seconds: int,
) -> str:
    """Build the full system prompt as a string.

    Parameters
    ----------
    game_state : dict
        Current crossword grid state.
    chat_history : str
        Full chat log (oldest → newest).  The assistant’s last message is
        expected to contain a JSON object with keys `strategy` and `message`.
    user_emotion : str
        One of {', '.join(sorted(set(FER_2013_EMO_DICT.values())))}.
    silence_seconds : int
        Seconds since the user last spoke.
    """

    # 1.  Incorrect entries block ------------------------------------------------
    error_msgs = _find_errors(game_state)
    incorrect_block = (
        "The user has at least one mistake:\n" + "\n".join(error_msgs)
        if error_msgs else ""
    )

    # 2.  Focal clue -------------------------------------------------------------
    focal_dir, focal_num = _choose_focal(game_state)
    focal_clue = a_CLUE_LOOKUP[(focal_dir, focal_num)]
    focal_pattern = game_state['across' if focal_dir == 'A' else 'down'].get(str(focal_num), "")
    focal_info = (
        f"The user’s cursor is at ({focal_dir}{focal_num}) “{focal_clue['hint']}”.\n"
        f"[INTERNAL] Correct answer (never reveal): {focal_clue['answer']}\n"
        f"Current pattern: “{_pattern_pretty(focal_pattern)}”"
    )

    # 3.  Other interesting clues ------------------------------------------------
    interesting_list = _pick_interesting(game_state, (focal_dir, focal_num))
    interesting_block = "\n".join(interesting_list) if interesting_list else ""

    # 4.  Rest‑of‑board snapshot -------------------------------------------------
    rest_snapshot = _summarise_rest(game_state, {(focal_dir, focal_num)})

    # 5.  Emotion block ----------------------------------------------------------
    emotion_block = f"Current user emotion detected: **{user_emotion}**"

    # ---------------------------------------------------------------------------
    # Compose the final system prompt
    # ---------------------------------------------------------------------------
    parts: List[str] = [_PROMPT_HEADER]

    if incorrect_block:
        parts.append("\n" + incorrect_block)

    parts.extend([
        "\n" + focal_info,
        "\n\nSome other clues you might bring up:\n" + interesting_block if interesting_block else "",
        "\n\nOther unsolved patterns (quick glance):\n" + rest_snapshot,
        f"\n\n### EMOTION\n{emotion_block}",
        f"\n\n### CHAT HISTORY (oldest → newest)\n(User silent for {silence_seconds} s)\n{chat_history}",
        "\n\n" + _PROMPT_FOOTER,
    ])

    return "".join(parts)
