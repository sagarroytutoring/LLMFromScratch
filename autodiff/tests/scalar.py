import unittest
import autodiff


class TestAutodiff(unittest.TestCase):

    def test_value(self):
        x = autodiff.Variable('x1')
        with autodiff.assign(x1=1):
            self.assertEqual(autodiff.value(x), 1)

    def test_const_addition_value(self):
        x = autodiff.Variable('x2')
        s = x + 3
        with autodiff.assign(x2=1):
            self.assertEqual(autodiff.value(s), 4)

    def test_const_addition_deriv(self):
        x = autodiff.Variable('x3')
        s = x + 3
        dsdx = autodiff.d(s, x)
        with autodiff.assign(x3=1):
            self.assertEqual(autodiff.value(dsdx), 1)

    def test_addition_value(self):
        x = autodiff.Variable('x4')
        y = autodiff.Variable('y1')
        s = x + y
        with autodiff.assign(x4=1, y1=2):
            self.assertEqual(autodiff.value(s), 3)

    def test_addition_deriv(self):
        x = autodiff.Variable('x5')
        y = autodiff.Variable('y2')
        s = x + y
        dsdx = autodiff.d(s, x)
        dsdy = autodiff.d(s, y)
        with autodiff.assign(x5=1, y2=2):
            self.assertEqual(autodiff.value(dsdx), 1)
            self.assertEqual(autodiff.value(dsdy), 1)

    def test_self_deriv(self):
        x = autodiff.Variable('x6')
        dxdx = autodiff.d(x, x)
        with autodiff.assign(x6=5):
            self.assertEqual(autodiff.value(dxdx), 1)

    def test_constant_mult_value(self):
        x = autodiff.Variable('x7')
        y = 2 * x
        with autodiff.assign(x7=3):
            self.assertEqual(autodiff.value(y), 6)

    def test_constant_mult_deriv(self):
        x = autodiff.Variable('x8')
        y = 2 * x
        dydx = autodiff.d(y, x)
        with autodiff.assign(x8=3):
            self.assertEqual(autodiff.value(dydx), 2)

    def test_mult_value(self):
        x = autodiff.Variable('x9')
        y = autodiff.Variable('y3')
        z = x * y
        with autodiff.assign(x9=3, y3=4):
            self.assertEqual(autodiff.value(z), 12)

    def test_mult_deriv(self):
        x = autodiff.Variable('x10')
        y = autodiff.Variable('y4')
        z = x * y
        dzdx = autodiff.d(z, x)
        dzdy = autodiff.d(y, x)
        with autodiff.assign(x10=3, y4=4):
            self.assertEqual(autodiff.value(dzdx), 4)
            self.assertEqual(autodiff.value(dzdy), 3)

    def test_power_value(self):
        x = autodiff.Variable('x11')
        sq = x**2
        with autodiff.assign(x11=4):
            self.assertEqual(autodiff.value(sq), 16)

    def test_power_deriv(self):
        x = autodiff.Variable('x12')
        sq = x**2
        dsqdx = autodiff.d(sq, x)
        with autodiff.assign(x12=4):
            self.assertEqual(autodiff.value(dsqdx), 8)

    def test_linear_combo(self):
        x = autodiff.Variable('x13')
        m = autodiff.Variable('m1')
        b = autodiff.Variable('b1')
        lin = m*x+b
        dlindx = autodiff.d(lin, x)
        dlindm = autodiff.d(lin, m)
        dlindb = autodiff.d(lin, b)
        with autodiff.assign(x13=5, m1=3, b1=2):
            self.assertEqual(autodiff.value(lin), 17)
            self.assertEqual(autodiff.value(dlindx), 3)
            self.assertEqual(autodiff.value(dlindm), 5)
            self.assertEqual(autodiff.value(dlindb), 1)

    def test_dist_formula(self):
        x = autodiff.Variable('x14')
        y = autodiff.Variable('y5')
        z = (x**2+y**2)**0.5
        dzdx = autodiff.d(z, x)
        dzdy = autodiff.d(z, y)
        with autodiff.assign(x14=3, y5=4):
            self.assertEqual(autodiff.value(z), 5)
            self.assertEqual(autodiff.value(dzdx), 3/5)
            self.assertEqual(autodiff.value(dzdy), 4/5)

    def test_mult_assignments(self):
        x = autodiff.Variable('x15')
        y = autodiff.Variable('y6')
        z = x + y
        with autodiff.assign(x15=3, y6=4):
            self.assertEqual(autodiff.value(z), 7)
        with autodiff.assign(x15=5, y6=3):
            self.assertEqual(autodiff.value(z), 8)

    def test_unassigned_error(self):
        x = autodiff.Variable('x16')
        y = autodiff.Variable('y7')
        z = x + y
        with self.assertRaises(autodiff.VariableAssignmentError):
            autodiff.value(z)

    def test_unassigned_after_assigned_error(self):
        x = autodiff.Variable('x17')
        y = autodiff.Variable('y8')
        z = x + y
        with autodiff.assign(x17=3, y8=4):
            self.assertEqual(autodiff.value(z), 7)

        with self.assertRaises(autodiff.VariableAssignmentError):
            autodiff.value(z)


if __name__ == '__main__':
    unittest.main()
