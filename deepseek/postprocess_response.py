import re
import json


def process_response(text):
    # Regex patterns:
    # 1. Pattern for JSON blocks delimited by ```json ... ```
    pattern_with_ticks = r"```json\s*(\{.*?\})\s*```"

    # 2. Pattern for JSON blocks without ```json delimiters.
    #    This will try to capture any substring that starts with { and ends with }.
    #    (Be cautious: this regex may overmatch in complex texts.)
    pattern_without_ticks = r"(\{[\s\S]*?\})"

    # Find matches using finditer so we can capture the starting index
    matches = []

    for m in re.finditer(pattern_with_ticks, text, re.DOTALL):
        matches.append((m.start(), m.group(1)))

    for m in re.finditer(pattern_without_ticks, text, re.DOTALL):
        # Exclude matches that are already captured by pattern_with_ticks.
        # Here we check that the match doesn't start with "```json" to avoid duplicates.
        if not text[max(0, m.start()-7):m.start()] == "```json":
            matches.append((m.start(), m.group(1)))

    # If any JSON blocks are found, sort them by their position and select the last one.
    if matches:
        matches.sort(key=lambda x: x[0])
        last_json_str = matches[-1][1]
        try:
            # Validate that the string is valid JSON
            data = json.loads(last_json_str)
            print(json.dumps(data, indent=2))
            return data
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Last JSON block content:", last_json_str)
            return {}
    else:
        print("No JSON blocks found.")
        return {}