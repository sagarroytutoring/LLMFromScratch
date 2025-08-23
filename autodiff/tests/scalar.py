import unittest
from autodiff import *
from math import e


class TestAutodiff(unittest.TestCase):

    def test_value(self):
        x = Variable('x1')
        with assign(x1=1):
            self.assertAlmostEqual(value(x), 1)

    def test_const_addition_value(self):
        x = Variable('x2')
        s = x + 3
        with assign(x2=1):
            self.assertAlmostEqual(value(s), 4)

    def test_const_addition_deriv(self):
        x = Variable('x3')
        s = x + 3
        dsdx = d(s, x)
        with assign(x3=1):
            self.assertAlmostEqual(value(dsdx), 1)

    def test_addition_value(self):
        x = Variable('x4')
        y = Variable('y1')
        s = x + y
        with assign(x4=1, y1=2):
            self.assertAlmostEqual(value(s), 3)

    def test_addition_deriv(self):
        x = Variable('x5')
        y = Variable('y2')
        s = x + y
        dsdx = d(s, x)
        dsdy = d(s, y)
        with assign(x5=1, y2=2):
            self.assertAlmostEqual(value(dsdx), 1)
            self.assertAlmostEqual(value(dsdy), 1)

    def test_self_deriv(self):
        x = Variable('x6')
        dxdx = d(x, x)
        with assign(x6=5):
            self.assertAlmostEqual(value(dxdx), 1)

    def test_constant_mult_value(self):
        x = Variable('x7')
        y = 2 * x
        with assign(x7=3):
            self.assertAlmostEqual(value(y), 6)

    def test_constant_mult_deriv(self):
        x = Variable('x8')
        y = 2 * x
        dydx = d(y, x)
        with assign(x8=3):
            self.assertAlmostEqual(value(dydx), 2)

    def test_mult_value(self):
        x = Variable('x9')
        y = Variable('y3')
        z = x * y
        with assign(x9=3, y3=4):
            self.assertAlmostEqual(value(z), 12)

    def test_mult_deriv(self):
        x = Variable('x10')
        y = Variable('y4')
        z = x * y
        dzdx = d(z, x)
        dzdy = d(z, y)
        with assign(x10=3, y4=4):
            self.assertAlmostEqual(value(dzdx), 4)
            self.assertAlmostEqual(value(dzdy), 3)

    def test_power_value(self):
        x = Variable('x11')
        sq = x**2
        with assign(x11=4):
            self.assertAlmostEqual(value(sq), 16)

    def test_power_deriv(self):
        x = Variable('x12')
        sq = x**2
        dsqdx = d(sq, x)
        with assign(x12=4):
            self.assertAlmostEqual(value(dsqdx), 8)

    def test_linear_combo(self):
        x = Variable('x13')
        m = Variable('m1')
        b = Variable('b1')
        lin = m*x+b
        dlindx = d(lin, x)
        dlindm = d(lin, m)
        dlindb = d(lin, b)
        with assign(x13=5, m1=3, b1=2):
            self.assertAlmostEqual(value(lin), 17)
            self.assertAlmostEqual(value(dlindx), 3)
            self.assertAlmostEqual(value(dlindm), 5)
            self.assertAlmostEqual(value(dlindb), 1)

    def test_dist_formula(self):
        x = Variable('x14')
        y = Variable('y5')
        z = (x**2+y**2)**0.5
        dzdx = d(z, x)
        dzdy = d(z, y)
        with assign(x14=3, y5=4):
            self.assertAlmostEqual(value(z), 5)
            self.assertAlmostEqual(value(dzdx), 3/5)
            self.assertAlmostEqual(value(dzdy), 4/5)

    def test_mult_assignments(self):
        x = Variable('x15')
        y = Variable('y6')
        z = x + y
        with assign(x15=3, y6=4):
            self.assertAlmostEqual(value(z), 7)
        with assign(x15=5, y6=3):
            self.assertAlmostEqual(value(z), 8)

    def test_unassigned_error(self):
        x = Variable('x16')
        y = Variable('y7')
        z = x + y
        with self.assertRaises(VariableAssignmentError):
            value(z)

    def test_compound_exp_unassigned_error(self):
        x = Variable('x18')
        y = Variable('y9')
        z = Variable('z1')
        w = z * (x + y)
        with self.assertRaises(VariableAssignmentError):
            value(w)

    def test_unassigned_after_assigned_error(self):
        x = Variable('x17')
        y = Variable('y8')
        z = x + y
        with assign(x17=3, y8=4):
            self.assertAlmostEqual(value(z), 7)

        with self.assertRaises(VariableAssignmentError):
            value(z)

    def test_compound_exp_unassigned_after_assigned_error(self):
        x = Variable('x19')
        y = Variable('y10')
        z = Variable('z2')
        w = z * (x + y)
        with assign(x19=1, y10=2, z2=3):
            self.assertAlmostEqual(value(w), 9)
        with self.assertRaises(VariableAssignmentError):
            value(w)

    def test_deriv_view_posthoc(self):
        x = Variable('x20')
        y = Variable('y11')
        z = x**2+y**2
        dzdx = d(z, x)
        with assign(x20=6, y11=4):
            self.assertAlmostEqual(value(z), 52)
            self.assertAlmostEqual(value(dzdx), 12)

            dzdy = d(z, y)
            self.assertAlmostEqual(value(dzdy), 8)

    def test_exp_value(self):
        x = Variable('x21')
        y = exp(x)
        with assign(x21=5):
            self.assertAlmostEqual(value(y), e**5)

    def test_exp_deriv(self):
        x = Variable('x21')
        y = exp(x)
        dydx = d(y, x)
        with assign(x21=5):
            self.assertAlmostEqual(value(dydx), e**5)


if __name__ == '__main__':
    unittest.main()
