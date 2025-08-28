import math
from typing import Any
from .basic_expressions import Expression, Variable, ExponentialExpression, LogExpression
from .advanced_expressions import LogisticExpression


class AssignmentContext:
    def __init__(self, assignments: dict[str, Any]):
        self._assignments = assignments

    def __enter__(self):
        for name, val in self._assignments.items():
            Variable.get_by_name(name).val = val

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self._assignments:
            Variable.get_by_name(name).unset()


class DerivativeView:
    def __init__(self, num: Expression, denom: Expression):
        self._num = num
        self._denom = denom

    @property
    def val(self):
        self._num._backprop()
        return self._denom._d[self._num]


def d(num, denom) -> DerivativeView:
    return DerivativeView(num, denom)


def assign(**kwargs):
    return AssignmentContext(kwargs)


def value(exp: Expression | DerivativeView):
    return exp.val


def exp(exponent):
    return ExponentialExpression(exponent) if isinstance(exponent, Expression) else math.exp(exponent)


def ln(arg):
    return LogExpression(arg) if isinstance(arg, Expression) else math.log(arg)


def log(arg, base=None):
    return ln(arg) if base is None else ln(arg) / ln(base)


def logistic(arg):
    return LogisticExpression(arg) if isinstance(arg, Expression) else 1 / (1 + math.exp(-arg))
