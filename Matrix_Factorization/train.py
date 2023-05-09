import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from _model import MF

def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    ratings_df = pd.read_csv("ratings.csv", encoding="utf-8")

    user_item_matrix = ratings_df.pivot_table("rating", "userId", "movieId").fillna(0)

    R = user_item_matrix.to_numpy()
    K = 20 #latent factor
    learning_rate = 0.01
    regularization = 0.2
    epochs = 30
    verbose = True

    mf = MF(R, K, learning_rate, regularization, epochs, verbose)
    mf.train()

    train_result_df = mf.training_process

    x = train_result_df.epoch.values
    y = train_result_df.rmse.values
    plt.plot(x, y)
    plt.xticks(x, x)
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.grid(axis="y")
    plt.show()

if __name__ =="__main__":
    main()