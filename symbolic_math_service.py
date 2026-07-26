#!/usr/bin/env python3

import ast
import re
from typing import Optional

import sympy


class SymbolicMathError(ValueError):
    """Raised when a recognized symbolic request is unsafe or unsupported."""


X = sympy.Symbol("x", positive=True)

_FUNCTIONS = {
    "log": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
}

_REQUEST_PATTERNS = (
    (
        "INTEGRAL",
        re.compile(
            r"^(?:find\s+)?(?:the\s+)?integral\s+of\s+(.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "INTEGRAL",
        re.compile(
            r"^what(?:\s+is|'s)\s+(?:the\s+)?integral\s+of\s+(.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "INTEGRAL",
        re.compile(r"^integrate\s+(.+)$", re.IGNORECASE),
    ),
    (
        "DERIVATIVE",
        re.compile(
            r"^(?:find\s+)?(?:the\s+)?derivative\s+of\s+(.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "DERIVATIVE",
        re.compile(
            r"^what(?:\s+is|'s)\s+(?:the\s+)?derivative\s+of\s+(.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "DERIVATIVE",
        re.compile(r"^differentiate\s+(.+)$", re.IGNORECASE),
    ),
)

_PHRASE_REPLACEMENTS = (
    (r"\b(?:the\s+)?natural\s+log(?:arithm)?\s+of\s+x\b", "log(x)"),
    (r"\b(?:the\s+)?natural\s+log(?:arithm)?\s+x\b", "log(x)"),
    (r"\bsine\s+of\s+x\b", "sin(x)"),
    (r"\bsine\s+x\b", "sin(x)"),
    (r"\bcosine\s+of\s+x\b", "cos(x)"),
    (r"\bcosine\s+x\b", "cos(x)"),
    (r"\btangent\s+of\s+x\b", "tan(x)"),
    (r"\btangent\s+x\b", "tan(x)"),
    (r"\bsquare\s+root\s+of\s+x\b", "sqrt(x)"),
    (r"\be\s+to\s+the\s+x\b", "exp(x)"),
    (r"\bx\s+squared\b", "x**2"),
    (r"\bx\s+cubed\b", "x**3"),
    (r"\bx\s+to\s+the\s+power\s+of\s+(-?\d+)\b", r"x**\1"),
    (r"\bmultiplied\s+by\b", "*"),
    (r"\bdivided\s+by\b", "/"),
    (r"\btimes\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
)

_MAX_INPUT_LENGTH = 160
_MAX_POWER = 10


def answer_symbolic_math_question(user_text: str) -> Optional[str]:
    """
    Return a spoken calculus answer, or None for non-calculus conversation.

    Parsing uses Python's AST only as a syntax tree. No eval, sympify,
    parse_expr, attribute access, subscripting, or arbitrary function calls
    are permitted.
    """
    if not isinstance(user_text, str):
        return None

    normalized = user_text.strip().rstrip("?. ")

    if not normalized:
        return None

    request = _extract_request(normalized)

    if request is None:
        return None

    operation, expression_text = request

    try:
        expression = _parse_expression(expression_text)

        if operation == "INTEGRAL":
            result = sympy.integrate(expression, X)

            if result.has(sympy.Integral):
                raise SymbolicMathError(
                    "SymPy could not evaluate the integral."
                )

            return (
                f"The integral is {_speak_expression(result)}, "
                "plus C."
            )

        result = sympy.diff(expression, X)

        if result.has(sympy.Derivative):
            raise SymbolicMathError(
                "SymPy could not evaluate the derivative."
            )

        return (
            f"The derivative is {_speak_expression(result)}."
        )

    except SymbolicMathError:
        return (
            "I can safely integrate or differentiate expressions in x "
            "using powers, natural log, sine, cosine, tangent, "
            "exponential, and square root."
        )


def _extract_request(user_text: str):
    for operation, pattern in _REQUEST_PATTERNS:
        match = pattern.fullmatch(user_text)

        if not match:
            continue

        expression = re.sub(
            r"\s+with\s+respect\s+to\s+x$",
            "",
            match.group(1).strip(),
            flags=re.IGNORECASE,
        )

        return operation, expression.strip()

    return None


def _normalize_expression(expression: str) -> str:
    if not expression or len(expression) > _MAX_INPUT_LENGTH:
        raise SymbolicMathError("Invalid symbolic expression length.")

    normalized = (
        expression.lower()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )

    for pattern, replacement in _PHRASE_REPLACEMENTS:
        normalized = re.sub(
            pattern,
            replacement,
            normalized,
            flags=re.IGNORECASE,
        )

    return " ".join(normalized.split())


def _parse_expression(expression: str):
    normalized = _normalize_expression(expression)

    if not re.fullmatch(
        r"[a-z0-9_+\-*/().\s]+",
        normalized,
    ):
        raise SymbolicMathError(
            "Expression contains unsupported characters."
        )

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise SymbolicMathError(
            "Invalid symbolic syntax."
        ) from exc

    return _build_expression(tree.body)


def _build_expression(node):
    if isinstance(node, ast.Constant):
        value = node.value

        if isinstance(value, bool):
            raise SymbolicMathError("Boolean values are not allowed.")

        if isinstance(value, int):
            return sympy.Integer(value)

        if isinstance(value, float):
            return sympy.Rational(str(value))

        raise SymbolicMathError("Only numeric constants are allowed.")

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise SymbolicMathError("Only the variable x is allowed.")

        return X

    if isinstance(node, ast.UnaryOp):
        operand = _build_expression(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand

        if isinstance(node.op, ast.USub):
            return -operand

        raise SymbolicMathError("Unary operator is not allowed.")

    if isinstance(node, ast.BinOp):
        left = _build_expression(node.left)
        right = _build_expression(node.right)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if isinstance(node.op, ast.Div):
            return left / right

        if isinstance(node.op, ast.Pow):
            if not right.is_Integer or abs(int(right)) > _MAX_POWER:
                raise SymbolicMathError(
                    "Power must be a small integer."
                )

            return left ** right

        raise SymbolicMathError("Binary operator is not allowed.")

    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or len(node.args) != 1
            or node.keywords
        ):
            raise SymbolicMathError(
                "Function call is not allowed."
            )

        return _FUNCTIONS[node.func.id](
            _build_expression(node.args[0])
        )

    raise SymbolicMathError(
        "Symbolic expression type is not allowed."
    )


def _speak_expression(expression) -> str:
    spoken = sympy.sstr(expression)

    replacements = (
        ("log(x)", "natural log of x"),
        ("sin(x)", "sine of x"),
        ("cos(x)", "cosine of x"),
        ("tan(x)", "tangent of x"),
        ("sqrt(x)", "square root of x"),
        ("exp(x)", "e to the x"),
        ("**", " to the power of "),
        ("*", " times "),
        ("/", " divided by "),
        (" + ", " plus "),
        (" - ", " minus "),
    )

    for old, new in replacements:
        spoken = spoken.replace(old, new)

    return " ".join(spoken.split())
