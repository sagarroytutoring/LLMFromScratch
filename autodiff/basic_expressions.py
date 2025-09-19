from typing import Optional, Any, Union
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import inspect
import math
from .exceptions import VariableAssignmentError


class Argument:
    def __init__(self):
        self._deriv = None
        self._pos: Optional[int] = None

    def derivative(self, func):
        self._deriv = func
        return func

    def __get__(self, instance: 'Expression', owner=None):
        return instance._subexps[self._pos]


class Expression(ABC):
    _derivs: list

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._derivs = []
        for pos, (name, argobj) in enumerate(inspect.getmembers(cls, lambda x: isinstance(x, Argument))):
            argobj._pos = pos
            cls._derivs.append(argobj._deriv)

    def __init__(self, *subexps: 'Expression'):
        self._subexps = subexps
        self._var_deps: set[Variable] = set()
        for exp in subexps:
            if isinstance(exp, Variable):
                exp._add_dependent_expression(self)
                self._var_deps.add(exp)
            else:
                self._var_deps.update(exp._var_deps)
                for var in exp._var_deps:
                    var._add_dependent_expression(self)
        self._val = None
        self._d: Counter['Expression', Any] = Counter()
        self.__cached_str: Optional[str] = None

    def __add__(self, other: Any) -> Union['AdditionExpression', 'ConstantAdditionExpression']:
        return AdditionExpression(self, other) if isinstance(other, Expression) else ConstantAdditionExpression(self, other)

    __radd__ = __add__

    def __mul__(self, other: Any) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        return MultiplicationExpression(self, other) if isinstance(other, Expression) else ConstantMultiplicationExpression(self, other)

    __rmul__ = __mul__

    def __pow__(self, power, modulo=None) -> Union['PowerExpression', 'ExponentialExpression']:
        if modulo is not None:
            raise NotImplementedError('Modular exponentiation is not implemented')
        return ExponentialExpression(LogExpression(self) * power) if isinstance(power, Expression) else PowerExpression(self, power)

    def __rpow__(self, other):
        return ExponentialExpression(math.log(other) * self)

    def __sub__(self, other) -> Union['AdditionExpression', 'ConstantAdditionExpression']:
        return self + -1 * other

    def __rsub__(self, other) -> Union['AdditionExpression', 'ConstantAdditionExpression']:
        return other + -1 * self

    def __truediv__(self, other) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        return self * other ** -1

    def __rtruediv__(self, other) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        return other * self ** -1

    def __neg__(self):
        return -1 * self

    @property
    def val(self):
        if self._val is None:
            self._val = self._get_value()
        return self._val

    @val.setter
    def val(self, v):
        self.set(v)

    def set(self, v):
        raise NotImplementedError('This expression does not allow value assignment')

    def unset(self):
        self._val = None
        self._d = Counter()

    @abstractmethod
    def _get_value(self):
        pass

    def _backprop(self) -> None:
        if self in self._d:  # There was already a backprop from this point
            return

        # Topo sort of all dependencies of self
        # Make the incoming list using DFS
        incoming = defaultdict(set)
        visited = set()
        q = [self]
        while q:
            curr = q.pop()
            for exp in curr._subexps:
                incoming[exp].add(curr)
                if exp not in visited:
                    q.append(exp)
                    visited.add(exp)

        # Kahns algo
        order = []
        free_nodes = [self]
        while free_nodes:
            curr_node = free_nodes.pop()
            order.append(curr_node)
            for nbr in curr_node._subexps:
                incoming[nbr].discard(curr_node)
                if not incoming[nbr]:
                    free_nodes.append(nbr)

        # Deriv of self wrt self is 1
        self._d[self] = 1

        # Loop through topo sorted deps and find derivs
        for exp in order:
            exp._deriv(self)

    def _deriv(self, numer: 'Expression') -> None:
        for idx, expr in enumerate(self._subexps):
            expr._d[numer] += self._d[numer] * self._derivs[idx](self)

    @property
    def _str(self):
        if self.__cached_str is None:
            self.__cached_str = self._make_str()
        return self.__cached_str

    def __str__(self) -> str:
        return f'{self._str}={self._val}'

    @abstractmethod
    def _make_str(self) -> str:
        pass


class Variable(Expression):
    _by_name: dict[str, 'Variable'] = {}

    @classmethod
    def get_by_name(cls, name: str) -> 'Variable':
        if name not in cls._by_name:
            raise KeyError(f'No Variable called {name} exists')
        return cls._by_name[name]

    def __init__(self, name: str):
        super().__init__()
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

    def _get_value(self):
        if not self._assigned:
            raise VariableAssignmentError(f'Variable {self._name} is not assigned')
        return self._val

    def _add_dependent_expression(self, exp: Expression):
        self._dependent_exps.add(exp)

    def _deriv(self, numer: 'Expression'):
        pass

    def _make_str(self) -> str:
        return self._name


class AdditionExpression(Expression):
    left_term = Argument()
    right_term = Argument()

    def _get_value(self):
        return self.left_term.val + self.right_term.val

    @left_term.derivative
    @right_term.derivative
    def deriv(self):
        return 1

    def _make_str(self) -> str:
        return f'({self.left_term._str} + {self.right_term._str})'


class ConstantAdditionExpression(Expression):
    exp_term = Argument()

    def __init__(self, exp_term: Expression, const_term: Any):
        super().__init__(exp_term)
        self._const = const_term

    def _get_value(self):
        return self.exp_term.val + self._const

    @exp_term.derivative
    def deriv(self):
        return 1

    def _make_str(self) -> str:
        return f'({self.exp_term._str} + {self._const})'


class MultiplicationExpression(Expression):
    left_factor = Argument()
    right_factor = Argument()

    def _get_value(self):
        return self.left_factor.val * self.right_factor.val

    @left_factor.derivative
    def left_deriv(self):
        return self.right_factor.val

    @right_factor.derivative
    def right_deriv(self):
        return self.left_factor.val

    def _make_str(self) -> str:
        return f'({self.left_factor._str} * {self.right_factor._str})'


class ConstantMultiplicationExpression(Expression):
    exp_factor = Argument()

    def __init__(self, exp_factor: Expression, const_factor: Any):
        super().__init__(exp_factor)
        self._const = const_factor

    def _get_value(self):
        return self.exp_factor.val * self._const

    @exp_factor.derivative
    def derivative(self):
        return self._const

    def _make_str(self) -> str:
        return f'({self._const} * {self.exp_factor._str})'


class PowerExpression(Expression):
    base = Argument()

    def __init__(self, base: Expression, power: Any):
        super().__init__(base)
        self._pow = power

    def _get_value(self):
        return self.base.val ** self._pow

    @base.derivative
    def derivative(self):
        return self._pow * self.base.val ** (self._pow - 1)

    def _make_str(self) -> str:
        return f'({self.base._str} ** {self._pow})'


class ExponentialExpression(Expression):
    exponent = Argument()

    def _get_value(self):
        return math.exp(self.exponent.val)

    @exponent.derivative
    def derivative(self):
        return self.val

    def _make_str(self) -> str:
        return f"exp({self.exponent._str})"


class LogExpression(Expression):
    arg = Argument()

    def _get_value(self):
        return math.log(self.arg.val)

    @arg.derivative
    def derivative(self):
        return 1 / self.arg.val

    def _make_str(self) -> str:
        return f"ln({self.arg._str})"
