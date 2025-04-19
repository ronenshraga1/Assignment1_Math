from math import isclose

import numpy as np

from asssigment_1 import ge, Secant_method


def test_ge(m1,m2):
  m1 = np.array(m1)
  m2 = np.array(m2)
  v1 = np.linalg.solve(m1[:,:-1], m1[:,-1])
  v2 = np.linalg.solve(m2[:,:-1], m2[:,-1])
  #assert (abs(v1 - v2)< 1e-6 ).all() ," ‼️matrixes not equivalent "
  print("PASS✅")
def print_mat(m):
    for l in m:
      print(l)


def run_tests():
    test_cases = [
        # 2 variables, 2 equations → 2x3
        (
            [[2, 1, 5],
             [1, -1, 1]],
            ge([[2, 1, 5],
                [1, -1, 1]])
        ),

        # 3 variables, 3 equations → 3x4
        (
            [[1, 2, -1, 3],
             [3, 2, 1, 10],
             [2, -1, 1, 2]],
            ge([[1, 2, -1, 3],
                [3, 2, 1, 10],
                [2, -1, 1, 2]])
        ),

        # Another valid 3x4 matrix
        (
            [[1, 1, 1, 6],
             [0, 2, 5, -4],
             [2, 5, -1, 27]],
            ge([[1, 1, 1, 6],
                [0, 2, 5, -4],
                [2, 5, -1, 27]])
        ),
    ]

    for i, (m_orig, m_ge) in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print("Original matrix:")
        print_mat(m_orig)
        print("After GE:")
        print_mat(m_ge)
        test_ge(m_orig, m_ge)
def run_custom_tests():
    test_cases = [
        # Test 1
        (
            [[2, 4, -2, 2],
             [4,9, -3, 8, ],
             [-2, -3, 7, -1]],

            [[1, 2, -1, 1],
             [0, 1, 1, 4],
             [0, 0, 1, -0.75]]
        ),

        # Test 2
        (
            [[4, 8, 12],
             [5, 16, 20]],

            [[1, 2, 3],
             [0, 1, 0.83333]]
        ),

        # Test 3
        (
            [[5, 8]],
            [[1, 1.6]]
        ),
    ]

    for i, (input_matrix, expected_output) in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print("Original matrix:")
        print_mat(input_matrix)

        print("Expected GE output:")
        print_mat(expected_output)

        actual_output = ge([row[:] for row in input_matrix])  # use copy

        print("Actual GE output:")
        print_mat(actual_output)

        test_ge(expected_output, actual_output)
def test_ge_valid_shape():
    test_cases = [
        # Square matrix: 3x3
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        # Duplicate rows: 2x2 square
        (
            [[1.0, 2.0], [0.0, 0.0]],
            [[1, 2], [1, 2]]
        ),
        # Needs row swap: 2x2
        (
            [[1.0, 2.0], [0.0, 1.0]],
            [[0, 2], [1, 2]]
        ),
        # Augmented: 3x2 (n = m + 1)
        (
            [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[1, 2], [2, 4], [0, 0]]
        ),
    ]

    def round_matrix(matrix, digits=4):
        return [[round(x, digits) for x in row] for row in matrix]

    passed = 0
    for i, (expected, input_matrix) in enumerate(test_cases):
        result = ge(input_matrix)
        result_rounded = round_matrix(result)
        expected_rounded = round_matrix(expected)
        if result_rounded == expected_rounded:
            print(f"Valid Shape Test {i+1}: ✅ Passed")
            passed += 1
        else:
            print(f"Valid Shape Test {i+1}: ❌ Failed")
            print("Expected:", expected_rounded)
            print("Got     :", result_rounded)

    print(f"\n{passed}/{len(test_cases)} valid shape tests passed.")


def test_secant():
    tests = [
        {
            "name": "sqrt(2)",
            "func": "x**2 - 2",
            "x0": 1, "x1": 2,
            "expected": 2 ** 0.5
        },
        {
            "name": "cube root of 27",
            "func": "x**3 - 27",
            "x0": 2, "x1": 4,
            "expected": 3
        },
        {
            "name": "x = cos(x)",
            "func": "x - cos(x)",
            "x0": 0.5, "x1": 1,
            "expected": 0.739085
        },
        {
            "name": "flat function at 0",
            "func": "x**3",
            "x0": -0.01, "x1": 0.01,
            "expected": 0
        },
        {
            "name": "division by zero",
            "func": "1",
            "x0": 1, "x1": 1,
            "expected": None
        },
        {
            "name": "multiple roots",
            "func": "x**3 - x",
            "x0": 0.5, "x1": 1,
            "expected": 1
        },{
    "name": "quadratic root at x=3",
    "func": "x**2 - 6*x + 9",  # (x-3)^2
            "x0": 1.0, "x1": 4.9,
            "expected": 3
},
{
    "name": "root of sine near 0",
    "func": "sin(x)",
    "x0": -1, "x1": 1,
    "expected": 0
},
{
    "name": "exponential equation e^x - 2 = 0",
    "func": "exp(x) - 2",
    "x0": 0, "x1": 1,
    "expected": 0.693147  # ln(2)
},
{
    "name": "logarithmic function log(x) = 0",
    "func": "log(x)",
    "x0": 0.5, "x1": 2,
    "expected": 1
},
{
    "name": "no root in range (positive function)",
    "func": "x**2 + 1",
    "x0": -2, "x1": 2,
    "expected": None
},
{
    "name": "multiple roots, close to 0",
    "func": "x * (x - 1) * (x + 1)",
    "x0": -0.5, "x1": 0.5,
    "expected": 0
},
{
    "name": "large initial guesses",
    "func": "x**2 - 4",
    "x0": 100, "x1": 50,
    "expected": 2  # Should converge eventually
}

    ]

    for test in tests:
        result = Secant_method(test["func"], test["x0"], test["x1"])
        expected = test["expected"]

        if expected is None:
            passed = result is None
        else:
            passed = result is not None and isclose(result, expected, rel_tol=1e-5, abs_tol=1e-4)

        print(f"Test '{test['name']}': {'✅ PASSED' if passed else '❌ FAILED'}")
        if not passed:
            print(f"  ➤ Got: {result}, Expected: {expected}")


# Run all tests
test_secant()
