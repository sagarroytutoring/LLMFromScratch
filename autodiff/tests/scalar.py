import unittest
from autodiff import *
import math


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
            self.assertAlmostEqual(value(y), math.exp(5))

    def test_exp_deriv(self):
        x = Variable('x34')
        y = exp(x)
        dydx = d(y, x)
        with assign(x34=5):
            self.assertAlmostEqual(value(dydx), math.exp(5))

    def test_const_sub_value(self):
        x = Variable('x22')
        y = x - 5
        with assign(x22=7):
            self.assertAlmostEqual(value(y), 2)

    def test_const_rsub_value(self):
        x = Variable('x25')
        y = 5 - x
        with assign(x25=7):
            self.assertAlmostEqual(value(y), -2)

    def test_const_sub_deriv(self):
        x = Variable('x24')
        y = x - 5
        dydx = d(y, x)
        with assign(x24=6):
            self.assertAlmostEqual(value(dydx), 1)

    def test_const_rsub_deriv(self):
        x = Variable('x26')
        y = 3 - x
        dydx = d(y, x)
        with assign(x26=6):
            self.assertAlmostEqual(value(dydx), -1)

    def test_sub_value(self):
        x = Variable('x23')
        y = Variable('y12')
        z = x - y
        with assign(x23=6, y12=3):
            self.assertAlmostEqual(value(z), 3)

    def test_sub_deriv(self):
        x = Variable('x27')
        y = Variable('y13')
        z = x - y
        dzdx = d(z, x)
        dzdy = d(z, y)
        with assign(x27=6, y13=3):
            self.assertAlmostEqual(value(dzdx), 1)
            self.assertAlmostEqual(value(dzdy), -1)

    def test_const_div_value(self):
        x = Variable('x28')
        y = x / 5
        with assign(x28=7):
            self.assertAlmostEqual(value(y), 7/5)

    def test_const_rdiv_value(self):
        x = Variable('x29')
        y = 5 / x
        with assign(x29=7):
            self.assertAlmostEqual(value(y), 5/7)

    def test_const_div_deriv(self):
        x = Variable('x30')
        y = x / 5
        dydx = d(y, x)
        with assign(x30=6):
            self.assertAlmostEqual(value(dydx), 1/5)

    def test_const_rdiv_deriv(self):
        x = Variable('x31')
        y = 3 / x
        dydx = d(y, x)
        with assign(x31=6):
            self.assertAlmostEqual(value(dydx), -1/12)

    def test_div_value(self):
        x = Variable('x32')
        y = Variable('y14')
        z = x / y
        with assign(x32=6, y14=3):
            self.assertAlmostEqual(value(z), 2)

    def test_div_deriv(self):
        x = Variable('x33')
        y = Variable('y15')
        z = x / y
        dzdx = d(z, x)
        dzdy = d(z, y)
        with assign(x33=6, y15=3):
            self.assertAlmostEqual(value(dzdx), 1/3)
            self.assertAlmostEqual(value(dzdy), -2/3)

    def test_ln_value(self):
        x = Variable('x35')
        y = ln(x)
        with assign(x35=10):
            self.assertAlmostEqual(value(y), math.log(10))

    def test_ln_deriv(self):
        x = Variable('x36')
        y = ln(x)
        dydx = d(y, x)
        with assign(x36=10):
            self.assertAlmostEqual(value(dydx), 1/10)

    def test_log_value(self):
        x = Variable('x37')
        y = log(x, base=2)
        with assign(x37=8):
            self.assertAlmostEqual(value(y), 3)

    def test_log_deriv(self):
        x = Variable('x38')
        y = log(x, base=2)
        dydx = d(y, x)
        with assign(x38=8):
            self.assertAlmostEqual(value(dydx), 1 / (8 * math.log(2)))

    def test_exp_base_value(self):
        x = Variable('x39')
        y = 2 ** x
        with assign(x39=4):
            self.assertAlmostEqual(value(y), 16)

    def test_exp_base_deriv(self):
        x = Variable('x40')
        y = 2 ** x
        dydx = d(y, x)
        with assign(x40=4):
            self.assertAlmostEqual(value(dydx), math.log(2)*16)

    def test_double_exp_value(self):
        x = Variable('x41')
        y = Variable('y16')
        z = x ** y
        with assign(x41=2, y16=4):
            self.assertAlmostEqual(value(z), 16)

    def test_double_exp_deriv(self):
        x = Variable('x42')
        y = Variable('y17')
        z = x ** y
        dzdx = d(z, x)
        dzdy = d(z, y)
        with assign(x42=3, y17=4):
            self.assertAlmostEqual(value(dzdx), 108)
            self.assertAlmostEqual(value(dzdy), math.log(3) * 81)


if __name__ == '__main__':
    unittest.main()
