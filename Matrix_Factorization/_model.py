import numpy as np
import pandas as pd
from typing import Tuple, List


class MF(object):
    def __init__(self, R, K, learning_rate, regularization, epochs, verbose=True):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.verbose = verbose
        
        self.training_process = list()
    
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size = (self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        self.samples = [
            (i, j, self.R[i,j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(self.samples)
            SGD(self.P, self.Q, self.b, self.b_u, self.b_i, self.samples, self.learning_rate, self.regularization)
            predicted_R = get_predicted_matrix(self.P, self.Q, self.b, self.b_u, self.b_i)
            rmse = get_rmse(self.R, predicted_R)
            self.training_process.append((epoch, rmse))

            if self.verbose and (epoch % 10 == 0):
                print("epoch: %d, error = %.4f" % (epoch, rmse))
        self.training_process = pd.DataFrame(self.training_process, columns = ['epoch', 'rmse'])

def SGD(
    P: np.ndarray,
    Q: np.ndarray,
    b: float,
    b_u: np.ndarray,
    b_i: np.ndarray,
    samples: List[Tuple],
    lr: float,
    regularization:float
) -> None:
    
    for user_id, item_id, rating in samples:
        predicted_rating = b + b_u[user_id] + b_i[item_id] + P[user_id, :].dot(Q[item_id, :].T)
        error = (rating - predicted_rating)
        
        b_u[user_id] += lr * (error - regularization * b_u[user_id])
        b_i[item_id] += lr * (error - regularization * b_i[item_id])

        P[user_id, :] += lr * (error * Q[item_id, :] - regularization * P[user_id, :])
        Q[item_id, :] += lr * (error * P[user_id, :] - regularization * Q[item_id, :])

def get_predicted_matrix(
    P: np.ndarray,
    Q: np.ndarray,
    b: float,
    b_u: np.ndarray,
    b_i: np.ndarray
) -> np.ndarray:
    
    if b is None:
        return P.dot(Q.T)
    else:
        return P.dot(Q.T) + b + b_u[:,np.newaxis] + b_i[np.newaxis,:]

def get_rmse(
    R: np.ndarray,
    predicted_R: np.ndarray
) -> float:

    user_idx, item_idx = R.nonzero() #0이 아닌 행, 열 idx
    error = list()

    for user_id, item_id in zip(user_idx, item_idx):
       square_error = (R[user_id, item_id] - predicted_R[user_id, item_id]) ** 2
       error.append(square_error)
    rmse = np.sqrt(np.asarray(error).mean())

    return rmse
