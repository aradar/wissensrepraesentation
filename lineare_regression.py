from typing import Callable, Union, List, Tuple

import numpy as np

HypothesisFunc = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
HypothesisCreatorFunc = Callable[[float, float], HypothesisFunc]
HypothesisCreatorVecFunc = Callable[[np.ndarray], HypothesisFunc]
LossFunc = Callable[[np.ndarray], np.ndarray]
LossCreatorFunc = Callable[[HypothesisCreatorVecFunc, np.ndarray, np.ndarray], LossFunc]
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


def logistic_hypothesis(thetas: np.ndarray) -> HypothesisFunc:
    def h(x_values: np.ndarray):
        # todo: check if this is correct
        #return logistic_function(np.matmul(thetas.transpose(), x_values))
        x_extended = extend_x_values(x_values)
        #return logistic_function(np.matmul(x_extended, thetas.transpose()))
        return logistic_function(np.matmul(thetas, x_extended.transpose()))

    return h


def cross_entropy_loss(hypothesis_creator: HypothesisCreatorVecFunc,
        x: np.ndarray, y: np.ndarray) -> LossFunc:
    def loss(theta: np.ndarray) -> np.ndarray:
        h = hypothesis_creator(theta)

        positive_loss = np.multiply(np.multiply(-1, y), np.log(h(x)))
        negative_loss = np.multiply(np.subtract(1, y), np.log(np.subtract(1, h(x))))

        return np.subtract(positive_loss, negative_loss)

    return loss


def squared_error_loss(hypothesis_creator: HypothesisCreatorVecFunc,
        x: np.ndarray, y: np.ndarray) -> LossFunc:
    def loss(theta: np.ndarray) -> np.ndarray:
        h = hypothesis_creator(theta)

        return np.multiply(1 / 2, np.power(np.subtract(h(x), y), 2))

    return loss


def cost_function(hypothesis_creator: HypothesisCreatorFunc, x: np.ndarray,
        y: np.ndarray) -> CostFunc:
    normalization = 1 / (2 * len(x))

    def j(theta0: float, theta1: float) -> float:
        h = hypothesis_creator(theta0, theta1)
        return normalization * np.power(h(x) - y, 2).sum()

    return j


def l2_regularization(thetas: np.ndarray, lambda_reg) -> float:
    return np.multiply(lambda_reg, np.square(thetas).sum())


def cost_function_vec(hypothesis_creator: HypothesisCreatorVecFunc,
        x: np.ndarray, y: np.ndarray,
        loss_creator: LossCreatorFunc = squared_error_loss,
        lambda_reg = 0.0) -> CostVecFunc:
    normalization = 1 / (2 * len(x))

    def j(theta: np.ndarray) -> float:
        loss_func = loss_creator(hypothesis_creator, x, y)
        regularization = l2_regularization(theta, lambda_reg)
        return np.multiply(normalization, np.add(loss_func(theta).sum(), regularization))

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


def compute_new_theta_vec(h: HypothesisFunc, x: np.ndarray, y: np.ndarray,
        theta: np.ndarray, learning_rate: float, lambda_reg: float) -> np.ndarray:
    m = len(x)
    extended_x = extend_x_values(x)
    shrink_factor = 1 - learning_rate * lambda_reg / m

    # todo: should we stop regulizing theta_0?

    new_theta = np.multiply(shrink_factor, theta) - learning_rate / m * extended_x.transpose().dot(h(x) - y)

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
        learning_rate: float, iterations: int = 10000,
        hypothesis_creator: HypothesisCreatorVecFunc = linear_hypothesis_vec,
        lambda_reg: float = 0.0,
        loss_creator: LossCreatorFunc = squared_error_loss) -> Tuple[List[float], np.ndarray]:
    j = cost_function_vec(hypothesis_creator, x, y, loss_creator)
    cost_history = []
    for i in range(iterations):
        cost_history.append(j(theta))
        h = hypothesis_creator(theta)
        theta = compute_new_theta_vec(h, x, y, theta, learning_rate, lambda_reg)

    cost_history.append(j(theta))

    return cost_history, theta


def feature_scaling(x_values: np.ndarray):
    x_scaled = np.copy(x_values)
    x_scaled = x_scaled.transpose()
    for j, feature_vec in enumerate(x_scaled):
        mean = np.mean(feature_vec)
        std = np.std(feature_vec)
        x_scaled[j] = np.multiply(np.subtract(feature_vec, mean), 1 / std)

    return x_scaled.transpose()


def logistic_function(values: np.ndarray) -> np.ndarray:
    return np.divide(1, np.add(1, np.exp(-values)))
