from .basic_expressions import Expression, Argument
import math


class LogisticExpression(Expression):
    arg = Argument()

    def _get_value(self):
        return 1 / (1 + math.exp(-self.arg.val))

    @arg.derivative
    def derivative(self):
        return self.val * (1 - self.val)

    def _make_str(self):
        return f"logistic({self.arg._str})"
