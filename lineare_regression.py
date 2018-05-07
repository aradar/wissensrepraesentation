from typing import Callable, Union, List, Tuple

import numpy as np

HypothesisFunc = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
HypothesisCreatorFunc = Callable[[float, float], HypothesisFunc]
CostFunc = Callable[[float, float], float]


def linear_hypothesis(theta_0: float, theta_1: float) -> HypothesisFunc:
    return lambda x: theta_0 + theta_1 * x


def cost_function(hypothesis_creator: HypothesisCreatorFunc, x: np.ndarray,
                  y: np.ndarray) -> CostFunc:
    normalization = 1 / (2 * len(x))

    def j(theta_0: float, theta_1: float):
        h = hypothesis_creator(theta_0, theta_1)
        return normalization * np.power(h(x) - y, 2).sum()

    return j


def compute_new_theta(x: np.ndarray, y: np.ndarray, theta_0: float,
                      theta_1: float, learning_rate: float) -> Tuple[float, float]:
    m = len(x)

    def theta_0_derivative(theta_0, theta_1):
        return 1 / m * (theta_0 + theta_1 * x - y).sum()

    def theta_1_derivative(theta_0, theta_1):
        return 1 / m * ((theta_0 + theta_1 * x - y) * x).sum()

    new_theta_0 = theta_0 - learning_rate * theta_0_derivative(theta_0, theta_1)
    new_theta_1 = theta_1 - learning_rate * theta_1_derivative(theta_0, theta_1)

    return new_theta_0, new_theta_1


def train_univariate_linear_regression(x: np.ndarray, y: np.ndarray, start_theta_0: float, start_theta_1: float, learning_rate: float,
                                       iterations: int = 10000) -> Tuple[List[float], float, float]:
    theta_0 = start_theta_0
    theta_1 = start_theta_1
    j = cost_function(linear_hypothesis, x, y)
    cost_history = []
    for i in range(iterations):
        theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, learning_rate)
        cost_history.append(j(theta_0, theta_1))

    return cost_history, theta_0, theta_1
