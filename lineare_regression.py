from typing import Callable, Union, List, Tuple

import numpy as np

HypothesisFunc = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
HypothesisCreatorFunc = Callable[[float, float], HypothesisFunc]
HypothesisCreatorVecFunc = Callable[[np.ndarray], HypothesisFunc]
CostFunc = Callable[[float, float], float]
CostVecFunc = Callable[[np.ndarray], float]


def extend_x_values(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def linear_hypothesis(theta_0: float, theta_1: float) -> HypothesisFunc:
    return lambda x: theta_0 + theta_1 * x


def linear_hypothesis_vec(thetas: np.ndarray) -> HypothesisFunc:
    def h(x_values: np.ndarray):
        x_extended = extend_x_values(x_values)
        return np.matmul(thetas, x_extended.transpose())

    return h


def cost_function(hypothesis_creator: HypothesisCreatorFunc, x: np.ndarray,
                  y: np.ndarray) -> CostFunc:
    normalization = 1 / (2 * len(x))

    def j(theta0: float, theta1: float):
        h = hypothesis_creator(theta0, theta1)
        return normalization * np.power(h(x) - y, 2).sum()

    return j


def cost_function_vec(hypothesis_creator: HypothesisCreatorVecFunc, x: np.ndarray,
                      y: np.ndarray) -> CostVecFunc:
    normalization = 1 / (2 * len(x))

    def j(theta: np.ndarray):
        h = hypothesis_creator(theta)
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


def compute_new_theta_vec(h: HypothesisFunc, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                          learning_rate: float) -> np.ndarray:
    m = len(x)
    extended_x = extend_x_values(x)

    new_theta = theta - learning_rate / m * extended_x.transpose().dot(h(x) - y)

    return new_theta


def gradient_descent(x: np.ndarray, y: np.ndarray, start_theta_0: float,
                     start_theta_1: float, learning_rate: float,
                     iterations: int = 10000) -> Tuple[List[float], float, float]:
    theta_0 = start_theta_0
    theta_1 = start_theta_1
    j = cost_function(linear_hypothesis, x, y)
    cost_history = []
    for i in range(iterations):
        cost_history.append(j(theta_0, theta_1))
        theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, learning_rate)

    cost_history.append(j(theta_0, theta_1))

    return cost_history, theta_0, theta_1


def gradient_descent_multi(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                           learning_rate: float,
                           iterations: int = 10000) -> Tuple[List[float], np.ndarray]:
    j = cost_function_vec(linear_hypothesis_vec, x, y)
    cost_history = []
    for i in range(iterations):
        cost_history.append(j(theta))
        h = linear_hypothesis_vec(theta)
        theta = compute_new_theta_vec(h, x, y, theta, learning_rate)

    cost_history.append(j(theta))

    return cost_history, theta


def feature_scaling(x_values: np.ndarray):
    x_scaled = np.copy(x_values)
    x_scaled = x_scaled.transpose()
    for j, feature_vec in enumerate(x_scaled):
        print(feature_vec.shape)
        mean = np.mean(feature_vec)
        std = np.std(feature_vec)
        x_scaled[j] = np.multiply(np.subtract(feature_vec, mean), 1 / std)

    return x_scaled.transpose()
