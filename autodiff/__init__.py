# What should be in this package
# - variables
# - scalar operations
#   - addition
#   - multiplication
# - activation functions
#   - relu
#   - sigmoid
#   - softmax
# - easy composition of functions
#   - e.g. z = (x + y)**2
# - easy derivatives
#   - e.g. dzdx = d(z, x) (derivative of z with respect to x)
# - easily assigning all free variables
#   - e.g. with assign(x=2, y=5)
#   - should I support partial assignment? like only assign x?
#       - then I could do something like x <= 2, but this is sort of jank
# - easily retrieving outputs and derivatives after assigning variables
#   - e.g. value(x)

# next phase:
# - matmul
# - vectorized operations
# - gradient?
# - update variables?

# bonus:
# - powers
# - exponentiation (e.g. 2^x)
# - log
# - trig functions
# - tanh
# - higher order derivatives maybe?


from abc import ABC, abstractmethod
from typing import Any


class VariableAssignmentError(BaseException): pass


class Expression(ABC):
    def __init__(self, subexps: tuple['Expression', ...]):
        self._subexps = subexps
        for exp in subexps:
            if isinstance(exp, Variable):
                exp._add_dependent_expression(self)
        self._val = None

    def __add__(self, other: 'Expression') -> 'AdditionExpression':
        return AdditionExpression(self, other)

    def __mul__(self, other: 'Expression') -> 'MultiplicationExpression':
        return MultiplicationExpression(self, other)

    @property
    def val(self):
        if self._val is None:
            self._val = self._calc()
        return self._val

    @val.setter
    def val(self, v):
        self.set(v)

    def set(self, v):
        raise NotImplementedError('This expression does not allow value assignment')

    def unset(self):
        self._val = None

    @abstractmethod
    def _calc(self):
        pass


class AdditionExpression(Expression):

    def __init__(self, left_term: Expression, right_term: Expression):
        super().__init__((left_term, right_term))
        self._left = left_term
        self._right = right_term

    def _calc(self):
        return self._left.val + self._right.val


class MultiplicationExpression(Expression):
    def __init__(self, left_factor: Expression, right_factor: Expression):
        super().__init__((left_factor, right_factor))
        self._left = left_factor
        self._right = right_factor

    def _calc(self):
        return self._left.val * self._right.val


class Variable(Expression):
    _by_name: dict[str, 'Variable'] = {}

    @classmethod
    def get_by_name(cls, name: str) -> 'Variable':
        if name not in cls._by_name:
            raise KeyError(f'No Variable called {name} exists')
        return cls._by_name[name]

    def __init__(self, name: str):
        super().__init__(tuple())
        if name in self._by_name:
            raise ValueError(f'Variable with name {name} already exists')
        self._by_name[name] = self
        self._dependent_exps: set[Expression] = set()
        self._name = name
        self._assigned = False
        self._val = None

    def set(self, v):
        if self._assigned:
            raise VariableAssignmentError(f'Variable {self._name} is already assigned')
        self._assigned = True
        self._val = v

    def unset(self):
        if not self._assigned:
            raise VariableAssignmentError(f'Variable {self._name} is not assigned')
        self._assigned = False
        super().unset()
        for exp in self._dependent_exps:
            exp.unset()

    def _calc(self):
        if not self._assigned:
            raise VariableAssignmentError(f'Variable {self._name} is not assigned')
        return self._val

    def _add_dependent_expression(self, exp: Expression):
        self._dependent_exps.add(exp)


class AssignmentContext:
    def __init__(self, assignments: dict[str, Any]):
        self._assignments = assignments

    def __enter__(self):
        for name, val in self._assignments.items():
            Variable.get_by_name(name).val = val

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self._assignments:
            Variable.get_by_name(name).unset()


def d(num, denom):
    pass


def assign(**kwargs):
    return AssignmentContext(kwargs)


def value(exp: Expression):
    return exp.val
