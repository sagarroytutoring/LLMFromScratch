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


class Expression:
    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass


class Variable(Expression):
    _by_name = {}

    def __init__(self, name: str):
        super().__init__()
        if name in self._by_name:
            raise ValueError(f'Variable with name {name} already exists')
        self._by_name[name] = self
        self._name = name


def d(num, denom):
    pass


def assign(**kwargs):
    pass


def value(exp):
    pass
