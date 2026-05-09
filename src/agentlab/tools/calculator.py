from __future__ import annotations

import ast
import operator
from typing import Any

from agentlab.tools.base import BaseTool


_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluate basic arithmetic expressions."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }

    def run(self, **kwargs: Any) -> float | int:
        expression = kwargs.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("'expression' must be a non-empty string")

        tree = ast.parse(expression, mode="eval")
        return _evaluate_ast(tree.body)


def _evaluate_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.BinOp):
        bin_op_type = type(node.op)
        operation = _BINARY_OPERATORS.get(bin_op_type)
        if operation is None:
            raise ValueError(f"Unsupported binary operator: {bin_op_type.__name__}")
        return operation(_evaluate_ast(node.left), _evaluate_ast(node.right))

    if isinstance(node, ast.UnaryOp):
        unary_op_type = type(node.op)
        operation = _UNARY_OPERATORS.get(unary_op_type)
        if operation is None:
            raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")
        return operation(_evaluate_ast(node.operand))

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    raise ValueError("Only numeric expressions are allowed")
