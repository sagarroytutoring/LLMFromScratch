# Done
# - variables
# - scalar operations
#   - addition
#   - multiplication
#   - subtraction
#   - division
# - easy composition of functions
#   - e.g. z = (x + y)**2
# - easy derivatives
#   - e.g. dzdx = d(z, x) (derivative of z with respect to x)
#   - however, weird thing to note is that other derivatives of z can be computed post hoc, but not other derivates of x or with respect to x
#       - other derivatives with respect to x can be computed post hoc if forward accumulation is used instead
#       - in the code I take advantage of this with caching
# - easily assigning all free variables
#   - e.g. with assign(x=2, y=5)
# - easily retrieving outputs and derivatives after assigning variables
#   - e.g. value(x)
# - powers
# - log

# What should be in this package
# - activation functions
#   - relu
#   - sigmoid
#   - softmax
# - should I support partial assignment? like only assign x?
#   - what additional features could this introduce? currying?
#   - then I could do something like x <= 2, but this is sort of jank

# next phase:
# - matmul
# - vectorized operations
# - gradient?
# - update variables e.g. gradient descent?

# bonus:
# - detect circular dependencies (should be pretty easy to add to kahns algo)
# - exponentiation (e.g. 2^x)
# - trig functions
# - tanh
# - higher order derivatives maybe?
# - forward accumulation


from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections import Counter, defaultdict
import math


class VariableAssignmentError(BaseException): pass


class Expression(ABC):
    def __init__(self, subexps: tuple['Expression', ...]):
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
        if isinstance(other, Expression):
            return AdditionExpression(self, other)
        else:
            return ConstantAdditionExpression(self, other)

    __radd__ = __add__

    def __mul__(self, other: Any) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        if isinstance(other, Expression):
            return MultiplicationExpression(self, other)
        else:
            return ConstantMultiplicationExpression(self, other)

    __rmul__ = __mul__

    def __pow__(self, power, modulo=None) -> 'PowerExpression':
        if modulo is not None:
            raise NotImplementedError('Modular exponentiation is not implemented')
        if isinstance(power, Expression):
            # TODO
            raise NotImplementedError()
        return PowerExpression(self, power)

    def __rpow__(self, other):
        # TODO
        pass

    def __sub__(self, other) -> Union['AdditionExpression', 'ConstantAdditionExpression']:
        return self + -1 * other

    def __rsub__(self, other) -> Union['AdditionExpression', 'ConstantAdditionExpression']:
        return other + -1 * self

    def __truediv__(self, other) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        return self * other ** -1

    def __rtruediv__(self, other) -> Union['MultiplicationExpression', 'ConstantMultiplicationExpression']:
        return other * self ** -1

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
        self._d = Counter()

    @abstractmethod
    def _calc(self):
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

    @abstractmethod
    def _deriv(self, numer: 'Expression') -> None:
        pass

    @property
    def _cached_str(self):
        if self.__cached_str is None:
            self.__cached_str = self._make_str()
        return self.__cached_str

    def __str__(self) -> str:
        return f'{self._cached_str}={self._val}'

    @abstractmethod
    def _make_str(self) -> str:
        pass


class AdditionExpression(Expression):
    def __init__(self, left_term: Expression, right_term: Expression):
        super().__init__((left_term, right_term))
        self._left = left_term
        self._right = right_term

    def _calc(self):
        return self._left.val + self._right.val

    def _deriv(self, numer: 'Expression') -> None:
        self._left._d[numer] += self._d[numer]
        self._right._d[numer] += self._d[numer]

    def _make_str(self) -> str:
        return f'({self._left._cached_str} + {self._right._cached_str})'


class ConstantAdditionExpression(Expression):
    def __init__(self, exp_term: Expression, const_term: Any):
        super().__init__((exp_term,))
        self._exp = exp_term
        self._const = const_term

    def _calc(self):
        return self._exp.val + self._const

    def _deriv(self, numer: 'Expression') -> None:
        self._exp._d[numer] += self._d[numer]

    def _make_str(self) -> str:
        return f'({self._exp._cached_str} + {self._const})'


class MultiplicationExpression(Expression):
    def __init__(self, left_factor: Expression, right_factor: Expression):
        super().__init__((left_factor, right_factor))
        self._left = left_factor
        self._right = right_factor

    def _calc(self):
        return self._left.val * self._right.val

    def _deriv(self, numer: 'Expression') -> None:
        self._left._d[numer] += self._right.val * self._d[numer]
        self._right._d[numer] += self._left.val * self._d[numer]

    def _make_str(self) -> str:
        return f'({self._left._cached_str} * {self._right._cached_str})'


class ConstantMultiplicationExpression(Expression):
    def __init__(self, exp_factor: Expression, const_factor: Any):
        super().__init__((exp_factor,))
        self._exp = exp_factor
        self._const = const_factor

    def _calc(self):
        return self._exp.val * self._const

    def _deriv(self, numer: 'Expression') -> None:
        self._exp._d[numer] += self._const * self._d[numer]

    def _make_str(self) -> str:
        return f'({self._const} * {self._exp._cached_str})'


class PowerExpression(Expression):
    def __init__(self, base: Expression, power: Any):
        super().__init__((base,))
        self._base = base
        self._pow = power

    def _calc(self):
        return self._base.val ** self._pow

    def _make_str(self) -> str:
        return f'({self._base._cached_str} ** {self._pow})'

    def _deriv(self, numer: 'Expression') -> None:
        self._base._d[numer] += self._d[numer] * self._pow * self._base.val ** (self._pow - 1)


class ExponentialExpression(Expression):
    def __init__(self, exponent: Expression):
        super().__init__((exponent,))
        self._expo = exponent

    def _calc(self):
        return math.exp(self._expo.val)

    def _deriv(self, numer: 'Expression') -> None:
        self._expo._d[numer] += self.val * self._d[numer]

    def _make_str(self) -> str:
        return f"exp({self._expo._cached_str})"


class LogExpression(Expression):
    def __init__(self, arg: Expression):
        super().__init__((arg,))
        self._arg = arg

    def _calc(self):
        return math.log(self._arg.val)

    def _deriv(self, numer: 'Expression') -> None:
        self._arg._d[numer] += self._d[numer] / self._arg.val

    def _make_str(self) -> str:
        return f"ln({self._arg._cached_str}"


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

    def _deriv(self, numer: 'Expression'):
        pass

    def _make_str(self) -> str:
        return self._name


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
    if isinstance(exponent, Expression):
        return ExponentialExpression(exponent)
    else:
        return math.exp(exponent)


def ln(arg):
    if isinstance(arg, Expression):
        return LogExpression(arg)
    else:
        return math.log(arg)


def log(arg, base=None):
    if base is None:
        return ln(arg)
    else:
        return ln(arg) / ln(base)
