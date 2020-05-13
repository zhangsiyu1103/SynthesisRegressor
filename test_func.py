import numpy as np

def test_func1(x):
    return np.exp(x**2+2*x+3)

def test_func2(x):
    return (np.log(2*x+1))**2 + 7

def test_func3(x):
    return np.exp(0.5*x + 2) + 5

def test_func4(x):
    return 1 / (np.log(6 * x + 4))

def test_func5(x):
    return 3 / (np.exp(-2*x - 8))

def test_func6(x):
    return 4 - 3/(np.log(x))

def test_func7(x):
    return (np.exp(3*x - 1))**3 - 40

def test_func8(x):
    return np.exp(0.3*x + 2) - np.exp(x - 2.5)

