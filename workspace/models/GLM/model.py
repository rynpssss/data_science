# encoding utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, data_raw):
        self.data_clean = data_raw

    def run(self):
        # EDAで安価な物件に外れ値が見受けられた
        # 下位1%をとりあえず除外とする（適当ではないが、正確でもない）
        THRESHOLD = 0.01
        self.exclude_outlier(THRESHOLD)

        # 上記以外の明らかな外れ値
        self.exclude_idx([524, 1299])

        # 正規分布に近づけ、線形回帰の精度を高める
        self.convert_log(["SalePrice"])

        # 多重共線性をなくす
        self.create_adding_column("AllSF", ["GrLivArea", "TotalBsmtSF"])
        self.create_adding_column("AllFlrsSF", ["1stFlrSF", "2ndFlrSF"])

    def exclude_outlier(self, THRESHOLD):
        low_row = round(self.data_clean.shape[0] * THRESHOLD)
        low_ids = self.data_clean.iloc[:low_row]
        low_ids = list(low_ids['Id'].unique())

        self.data_clean = self.data_clean.query("Id not in @low_ids")

    def exclude_idx(self, ids):
        self.data_clean = self.data_clean.query("Id not in @ids")

    def convert_log(self, columns):
        for c in columns:
            self.data_clean[c] = self.data_clean[c].apply(lambda x: np.log(x))

    def create_adding_column(self, create, adding):
        c1, c2 = adding
        self.data_clean[create] = self.data_clean[c1] + self.data_clean[c2]


class Glm:
    def __init__(self, preprocessing, X_columns, y_column):
        self.X = preprocessing.data_clean[X_columns]
        self.y = preprocessing.data_clean[y_column]

    def fit(self):
        TRAIN_SIZE = 0.8 # >=0.7 なら自由
        RANDOM_STATE = 0 # チューニングはしていない
        x_train, x_test, y_train, y_test = \
                        self.train_test_split(TRAIN_SIZE, RANDOM_STATE)
    
        x_train, x_test = self.normalization(x_train, x_test)

        self.model = sm.OLS(y_train, sm.add_constant(x_train))
        self.model = self.model.fit()

    def train_test_split(self, TRAIN_SIZE, RANDOM_STATE):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                    train_size=TRAIN_SIZE, 
                                                    random_state=RANDOM_STATE)
        return x_train, x_test, y_train, y_test

    def normalization(self, x_train, x_test):
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)    
        x_test = scaler.transform(x_test)

        return x_train, x_test
    
    def write_summary(self, write_path):
        with open(write_path, "w") as f:
            f.write(str(self.model.summary()))


def main():
    data_raw = pd.read_csv("./../../data/house_prices/train.csv")

    preprocessing = Preprocessing(data_raw)
    preprocessing.run()

    X_columns = ["OverallQual", "GarageArea", "YearBuilt", "AllSF", 
                 "AllFlrsSF", "YearRemodAdd", "OverallCond"]
    y_column = ["SalePrice"]
    model = Glm(preprocessing, X_columns, y_column)
    model.fit()
    
    model.write_summary("./GLM_summary.txt")

if __name__ == "__main__":
    main()