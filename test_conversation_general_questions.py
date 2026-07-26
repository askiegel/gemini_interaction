#!/usr/bin/env python3

from pathlib import Path


def main():
    prompt = Path("prompts.py").read_text()

    required_rules = [
        "general knowledge, educational, and straightforward arithmetic",
        "Perform the calculation when needed.",
        "Do not claim that the robot is not designed for math",
        "concise and natural for spoken output",
    ]

    for rule in required_rules:
        if rule not in prompt:
            raise AssertionError(
                f"Conversation prompt is missing rule: {rule}"
            )

        print(f"PASS: conversation prompt contains: {rule}")

    print()
    print("General conversational question prompt test passed.")


if __name__ == "__main__":
    main()
