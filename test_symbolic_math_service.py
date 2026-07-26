#!/usr/bin/env python3

from symbolic_math_service import answer_symbolic_math_question


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected!r}\nActual: {actual!r}"
        )

    print(f"PASS: {message}")


def main():
    cases = [
        (
            "Find the integral of natural log of x.",
            (
                "The integral is x times natural log of x "
                "minus x, plus C."
            ),
        ),
        (
            "What's the integral of the natural log of x.",
            (
                "The integral is x times natural log of x "
                "minus x, plus C."
            ),
        ),
        (
            "What is integral of natural log of x.",
            (
                "The integral is x times natural log of x "
                "minus x, plus C."
            ),
        ),
        (
            "Integrate x squared.",
            (
                "The integral is x to the power of 3 "
                "divided by 3, plus C."
            ),
        ),
        (
            "Differentiate sine of x.",
            "The derivative is cosine of x.",
        ),
        (
            "Find the derivative of x cubed.",
            (
                "The derivative is 3 times x to the power of 2."
            ),
        ),
        (
            "What is the derivative of natural log of x?",
            "The derivative is 1 divided by x.",
        ),
    ]

    for question, expected in cases:
        assert_equal(
            answer_symbolic_math_question(question),
            expected,
            f"symbolic calculus works for {question}",
        )

    assert_equal(
        answer_symbolic_math_question("What planet is Mars?"),
        None,
        "non-calculus conversation is left for Gemini",
    )

    safe_failure = (
        "I can safely integrate or differentiate expressions in x "
        "using powers, natural log, sine, cosine, tangent, "
        "exponential, and square root."
    )

    unsafe_cases = [
        "Integrate __import__('os').system('echo unsafe').",
        "Differentiate open(x).",
        "Integrate x.__class__.",
        "Integrate y squared.",
        "Integrate x to the power of 100.",
    ]

    for question in unsafe_cases:
        assert_equal(
            answer_symbolic_math_question(question),
            safe_failure,
            f"unsafe symbolic input is rejected: {question}",
        )

    print()
    print("Restricted symbolic math service test passed.")


if __name__ == "__main__":
    main()
