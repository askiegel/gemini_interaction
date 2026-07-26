#!/usr/bin/env python3

import ast
import math
import operator
import re
from typing import Optional


class ArithmeticQuestionError(ValueError):
    """Raised when recognized arithmetic cannot be evaluated safely."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_QUESTION_PATTERNS = (
    re.compile(r"^what\s+is\s+(.+?)[?]?$", re.IGNORECASE),
    re.compile(r"^what(?:'s|\s+is)\s+(.+?)[?]?$", re.IGNORECASE),
    re.compile(r"^calculate\s+(.+?)[?]?$", re.IGNORECASE),
    re.compile(r"^compute\s+(.+?)[?]?$", re.IGNORECASE),
    re.compile(r"^how\s+much\s+is\s+(.+?)[?]?$", re.IGNORECASE),
)

_WORD_OPERATORS = (
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
)

_MAX_EXPRESSION_LENGTH = 100
_MAX_ABSOLUTE_VALUE = 1_000_000_000_000


def answer_arithmetic_question(user_text: str) -> Optional[str]:
    """
    Return a spoken arithmetic answer, or None when text is not a math question.

    Only numeric literals, parentheses, addition, subtraction,
    multiplication, division, and unary signs are accepted.
    """
    if not isinstance(user_text, str):
        return None

    normalized = user_text.strip()

    if not normalized:
        return None

    expression = _extract_expression(normalized)

    if expression is None:
        return None

    try:
        value = _evaluate_expression(expression)
    except ZeroDivisionError:
        return "I cannot divide by zero."
    except ArithmeticQuestionError:
        return (
            "I can calculate numbers using addition, subtraction, "
            "multiplication, division, and parentheses."
        )

    return f"The answer is {_format_number(value)}."


def _extract_expression(user_text: str) -> Optional[str]:
    expression = None

    for pattern in _QUESTION_PATTERNS:
        match = pattern.fullmatch(user_text)

        if match:
            expression = match.group(1).strip()
            break

    if expression is None:
        return None

    expression = (
        expression
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )

    for pattern, replacement in _WORD_OPERATORS:
        expression = pattern.sub(replacement, expression)

    return expression.strip()


def _evaluate_expression(expression: str):
    if not expression or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ArithmeticQuestionError("Invalid expression length.")

    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ArithmeticQuestionError("Expression contains unsupported input.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticQuestionError("Invalid arithmetic syntax.") from exc

    value = _evaluate_node(tree.body)

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArithmeticQuestionError("Arithmetic result is not finite.")

    if abs(value) > _MAX_ABSOLUTE_VALUE:
        raise ArithmeticQuestionError("Arithmetic result is too large.")

    return value


def _evaluate_node(node):
    if isinstance(node, ast.Constant):
        value = node.value

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArithmeticQuestionError("Only numbers are allowed.")

        if abs(value) > _MAX_ABSOLUTE_VALUE:
            raise ArithmeticQuestionError("Number is too large.")

        return value

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ArithmeticQuestionError("Operator is not supported.")

        return operation(
            _evaluate_node(node.left),
            _evaluate_node(node.right),
        )

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ArithmeticQuestionError("Unary operator is not supported.")

        return operation(_evaluate_node(node.operand))

    raise ArithmeticQuestionError("Expression type is not supported.")


def _format_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    if isinstance(value, int):
        return str(value)

    return format(value, ".10g")
