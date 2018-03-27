
import numpy as np
import pandas as pd

def get_normalized_data():

    print("Reading in and transforming data...")
    print("dd")

    # if not os.path.exists('../../data/minst/train.csv'):
    #     print('Looking for ../large_files/train.csv')
    #     print('You have not downloaded the data and/or not placed the files in the correct location.')
    #     print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
    #     print('Place train.csv in the folder large_files adjacent to the class folder')
    #     exit()

    df = pd.read_csv('../data/minst/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    # X is a matrix of all the samples excluding the first column which is label column
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std # normalize the data
    Y = data[:, 0]
    return X, Y
