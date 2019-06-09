import numpy as np

'''
Kernels for Support Vector Machines
'''

def linear_kernel(**kwargs):
    ''' Basic kernel '''
    def f(x1,x2):
        return np.inner(x1,x2)
    return f

def polynomial_kernel(power,coef,**kwargs):
    ''' https://en.wikipedia.org/wiki/Polynomial_kernel '''
    def f(x1,x2):
        return (np.inner(x1,x2) + coef)**power
    return f

def rbf_kernel(gamma,**kwargs):
    ''' https://en.wikipedia.org/wiki/Radial_basis_function_kernel '''
    def f(x1,x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)

    return f

