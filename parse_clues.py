def parse_clues(file_path):
    """
    Parse a crossword‐style .txt file into a list of clue dictionaries.

    Expected file structure (order matters):
      exolve-across:
        <number> <hint>
        ...
      exolve-down:
        <number> <hint>
        ...
      
      exolve-across:
        <number> <ANSWER>
        ...
      exolve-down:
        <number> <ANSWER>
        ...

    Returns:
        List[dict], where each dict has keys:
          - 'direction':  'across' or 'down'
          - 'number':     int
          - 'hint':       str
          - 'answer':     str
    """
    hints = {}
    answers = {}
    hints_seen = {'across': False, 'down': False}
    answers_seen = {'across': False, 'down': False}
    current_section = None  # tuple of ('hint' or 'answer', 'across' or 'down')

    with open(file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # skip blank lines
                continue

            low = line.lower()
            if low.startswith("exolve-across"):
                # First time: this begins the "hints across" block.
                # Second time: this begins the "answers across" block.
                if not hints_seen['across']:
                    current_section = ("hint", "across")
                    hints_seen['across'] = True
                else:
                    current_section = ("answer", "across")
                    answers_seen['across'] = True
                continue

            if low.startswith("exolve-down"):
                # First time: this begins the "hints down" block.
                # Second time: this begins the "answers down" block.
                if not hints_seen['down']:
                    current_section = ("hint", "down")
                    hints_seen['down'] = True
                else:
                    current_section = ("answer", "down")
                    answers_seen['down'] = True
                continue

            # If we're inside a known section, parse "<number> <text>"
            if current_section is not None:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    # Malformed line; skip
                    continue

                num_str, rest = parts
                try:
                    num = int(num_str)
                except ValueError:
                    # Not starting with an integer—skip
                    continue

                kind, direction = current_section  # e.g. ("hint", "across")
                if kind == "hint":
                    hints[(direction, num)] = rest
                else:
                    answers[(direction, num)] = rest

    # Combine hints + answers into a list of dicts
    clue_list = []
    for (direction, num), hint_text in hints.items():
        answer_text = answers.get((direction, num), "")
        clue_list.append({
            "direction": direction,
            "number": num,
            "hint": hint_text,
            "answer": answer_text
        })

    # Sort by direction then number
    clue_list.sort(key=lambda c: (c["direction"], c["number"]))
    return clue_list


def test_parse_clues():
    """
    Test function: load a sample .txt and print the full clue_list
    """
    # Replace 'crossword.txt' with your actual filename
    filename = "crossword_a.txt"
    try:
        clues = parse_clues(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    print(clues)


if __name__ == "__main__":
    test_parse_clues()
