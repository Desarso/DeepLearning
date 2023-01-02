import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List
import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline



# A function that takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def chain_length_2(chain: Chain, a: ndarray) -> ndarray:

    assert len(chain) == 2, \
        "chain_length_2 requires a chain of length 2"
    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(a))

def square(x: ndarray) -> ndarray:

    # square each element of the input array
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:

    # apply the leaky relu function to each element of the input array
    return np.maximum(0.2*x, x)

def deriv(func: Callable[[ndarray], ndarray],
         input_: ndarray,
         delta: float = 0.001) -> ndarray:
    
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def sigmoid(x: ndarray) -> ndarray:
    
        return 1 / (1 + np.exp(-x))

def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    
    assert len(chain) == 2, \
        "This function requires a 'Chain' objects of length 2"
    
    assert input_range.ndim == 1, \
        "Function requires a 1D array as input_range"
    
    f1 = chain[0]
    f2 = chain[1]

    #df1/dx
    f1_of_x = f1(input_range)

    #df1/du
    df1dx = deriv(f1, input_range)

    #df2/du(f1(x))
    df2du = deriv(f2, f1_of_x)

    return df2du * df1dx

def plot_chain(ax, chain: Chain, input_range: ndarray, length: int=2) -> None:
    assert input_range.ndim == 1, \
        "Function requires a 1D  as input_range"

    if length == 2:
        output_range = chain_length_2(chain, input_range)
    elif length == 3:
        output_range = chain_length_3(chain, input_range)
    ax.plot(input_range, output_range)

def plot_chain_deriv(ax, chain: Chain, input_range: ndarray, length: int=2) -> ndarray:
    
    if length == 2:
        output_range = chain_deriv_2(chain, input_range)
    elif length == 3:
        output_range = chain_deriv_3(chain, input_range)
    ax.plot(input_range, output_range)

def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 3, \
        "This function requires a 'Chain' objects of length 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    #f1(x)
    f1_of_x = f1(input_range)
    
    #f2(f1(x))
    f2_of_x = f2(f1_of_x)
    
    #df3du
    df3du = deriv(f3, f2_of_x)
    
    #df2du
    df2du = deriv(f2, f1_of_x)
    
    df1dx = deriv(f1, input_range)
    
    #Multiplying these quantities together at each point
    return df1dx * df2du * df3du

def chain_length_3(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) == 3, \
        "chain_length_3 requires a chain of length 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    return f3(f2(f1(a)))

def multiple_inputs_add(x: ndarray, y: ndarray, sigma: Array_Function) -> float:
    assert x.shape == y.shape, \
        "x and y must have the same shape"
    
    return sigma(x + y)

def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma: Array_Function) -> float:
    a = x + y   
    
    dsda = deriv(sigma, a)
    
    dadx, dady = 1, 1
    
    return dsda * dadx, dsda * dady
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## stringing two functions example
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))


# chain_1 = [square, sigmoid]
# chain_2 = [sigmoid, square]

# PLOT_RANGE = np.arange(-3, 3, 0.01)
# plot_chain(ax[0], chain_1, PLOT_RANGE)
# plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

# ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
# ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")

# plot_chain(ax[1], chain_2, PLOT_RANGE)
# plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)
# ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
# ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(x))$")

# plt.savefig("chain_deriv.png")

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## THREE FUNCTION STRIGGING EXAMPLE

# fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))  # 2 Rows, 1 Col

# chain_1 = [leaky_relu, square, sigmoid]
# chain_2 = [leaky_relu, sigmoid, square]

# PLOT_RANGE = np.arange(-3, 3, 0.01)
# plot_chain(ax[0], chain_1, PLOT_RANGE, length=3)
# plot_chain_deriv(ax[0], chain_1, PLOT_RANGE, length=3)

# ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
# ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(leakyRrelu(x)))$")

# plot_chain(ax[1], chain_2, PLOT_RANGE, length=3)
# plot_chain_deriv(ax[1], chain_2, PLOT_RANGE, length=3)
# ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
# ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(leakyRelu(x)))$")

# plt.savefig("09_plot_chain_rule2.png")

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

