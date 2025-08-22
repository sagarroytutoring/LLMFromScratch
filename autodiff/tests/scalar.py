import unittest
import autodiff


class TestAutodiff(unittest.TestCase):

    def test_value(self):
        x = autodiff.Variable('x')
        with autodiff.assign(x=1):
            self.assertEquals(autodiff.value(x), 1)

    def test_const_addition_value(self):
        x = autodiff.Variable('x')
        s = x + 3
        with autodiff.assign(x=1):
            self.assertEquals(autodiff.value(s), 4)

    def test_const_addition_deriv(self):
        x = autodiff.Variable('x')
        s = x + 3
        dsdx = autodiff.d(s, x)
        with autodiff.assign(x=1):
            self.assertEquals(autodiff.value(dsdx), 1)

    def test_addition_value(self):
        x = autodiff.Variable('x')
        y = autodiff.Variable('y')
        s = x + y
        with autodiff.assign(x=1, y=2):
            self.assertEquals(autodiff.value(s), 3)

    def test_addition_deriv(self):
        x = autodiff.Variable('x')
        y = autodiff.Variable('y')
        s = x + y
        dsdx = autodiff.d(s, x)
        dsdy = autodiff.d(s, y)
        with autodiff.assign(x=1, y=2):
            self.assertEquals(autodiff.value(dsdx), 1)
            self.assertEquals(autodiff.value(dsdy), 1)

    def test_self_deriv(self):
        x = autodiff.Variable('x')
        dxdx = autodiff.d(x, x)
        with autodiff.assign(x=5):
            self.assertEquals(autodiff.value(dxdx), 1)

    def test_constant_mult_value(self):
        x = autodiff.Variable('x')
        y = 2 * x
        with autodiff.assign(x=3):
            self.assertEquals(autodiff.value(y), 6)

    def test_constant_mult_deriv(self):
        x = autodiff.Variable('x')
        y = 2 * x
        dydx = autodiff.d(y, x)
        with autodiff.assign(x=3):
            self.assertEquals(autodiff.value(dydx), 2)

    def test_mult_value(self):
        x = autodiff.Variable('x')
        y = autodiff.Variable('y')
        z = x * y
        with autodiff.assign(x=3, y=4):
            self.assertEquals(autodiff.value(z), 12)

    def test_mult_deriv(self):
        x = autodiff.Variable('x')
        y = autodiff.Variable('y')
        z = x * y
        dzdx = autodiff.d(z, x)
        dzdy = autodiff.d(y, x)
        with autodiff.assign(x=3, y=4):
            self.assertEquals(autodiff.value(dzdx), 4)
            self.assertEquals(autodiff.value(dzdy), 3)

    def test_power_value(self):
        x = autodiff.Variable('x')
        sq = x**2
        with autodiff.assign(x=4):
            self.assertEquals(autodiff.value(sq), 16)

    def test_power_deriv(self):
        x = autodiff.Variable('x')
        sq = x**2
        dsqdx = autodiff.d(sq, x)
        with autodiff.assign(x=4):
            self.assertEquals(autodiff.value(dsqdx), 8)

    def test_linear_combo(self):
        x = autodiff.Variable('x')
        m = autodiff.Variable('m')
        b = autodiff.Variable('b')
        lin = m*x+b
        dlindx = autodiff.d(lin, x)
        dlindm = autodiff.d(lin, m)
        dlindb = autodiff.d(lin, b)
        with autodiff.assign(x=5, m=3, b=2):
            self.assertEquals(autodiff.value(lin), 17)
            self.assertEquals(autodiff.value(dlindx), 3)
            self.assertEquals(autodiff.value(dlindm), 5)
            self.assertEquals(autodiff.value(dlindb), 1)

    def test_dist_formula(self):
        x = autodiff.Variable('x')
        y = autodiff.Variable('y')
        z = (x**2+y**2)**0.5
        dzdx = autodiff.d(z, x)
        dzdy = autodiff.d(z, y)
        with autodiff.assign(x=3, y=4):
            self.assertEquals(autodiff.value(z), 5)
            self.assertEquals(autodiff.value(dzdx), 3/5)
            self.assertEquals(autodiff.value(dzdy), 4/5)


if __name__ == '__main__':
    unittest.main()
