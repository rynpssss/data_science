# {~/data_science_refercterring/workspace/notebooks/Logistice_Regression/logistic_model.ipynb}
# をリファクタリングしたスクリプトファイル
# リーダブルコードを読んで、できるだけ読みやすいプログラムを目指した（22.04.18)

# coding utf-8

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm

class Preprocessing:
    def __init__(self, train, test):
        self.train_data = train
        self.test_data = test

    def run(self) -> None:
        # 精度向上に寄与しない説明変数のため削除
        self.drop_column(['Name', 'Ticket', 'Cabin', 'Embarked'])

        # ラベルエンコーディングを選択した意味は特にない
        # 線形回帰のため、単純な数値変換が行いたかった
        columns = list(self.train_data.select_dtypes(include="O").columns)
        self.label_encoding(columns)

        # EDAから効果的な値埋めを採用
        self.fillna_ave_by_group(fillna_column="Age", group=["Pclass", "Sex"])

        # 精度に最も効果的だった方法を採用
        columns = list(self.train_data
                      .drop(["PassengerId", "Survived"], axis=1).columns)
        self.normalization(columns)

    def drop_column(self, columns):
        self.train_data = self.train_data.drop(columns, axis=1)
        self.test_data = self.test_data.drop(columns, axis=1)

    def label_encoding(self, columns):
        for c in columns:
            le = LabelEncoder()
            le.fit(self.train_data[c])
            self.train_data[c] = le.transform(self.train_data[c])
            self.test_data[c] = le.transform(self.test_data[c])

    def fillna_ave_by_group(self, fillna_column, group):
        self.train_data[fillna_column] = \
                        self.train_data.groupby(group)[fillna_column]\
                                       .apply(lambda x: x.fillna(x.mean()))
        self.test_data[fillna_column] = \
                        self.test_data.groupby(group)[fillna_column]\
                                      .apply(lambda x: x.fillna(x.mean()))

    def normalization(self, columns):
        scaler = StandardScaler()
        scaler.fit(self.train_data[columns])

        self.train_data[columns] = scaler.transform(self.train_data[columns])
        self.test_data[columns] = scaler.transform(self.test_data[columns])


class LogisticRegression:
    def __init__(self, preprocessing):
        self.train_data = preprocessing.train_data
        self.test_data = preprocessing.test_data

    def fit(self, formula):
        self.model = smf.glm(formula=formula, 
                             data=self.train_data, 
                             family=sm.families.Binomial())\
                            .fit()

    def predict(self):
        pred = self.model.predict(self.test_data).to_frame(name="Survived")
        pred['Survived'] = pred['Survived'].apply(lambda x: round(x))

        # Idを合わせる
        START_ID_IN_TEST_DATA = 892
        self.pred = pred.reset_index()
        self.pred = self.pred.rename(columns={'index':'PassengerId'})
        self.pred['PassengerId'] = self.pred['PassengerId']\
                                    .apply(lambda x: x + START_ID_IN_TEST_DATA)

    def write_summary(self, save_path):
        with open(save_path, "w")as f:
            f.write(str(self.model.summary()))

    def write_predict(self, save_path):
        self.pred.to_csv(save_path, index=False, encoding='utf-8-sig')


def main():
    train_raw = pd.read_csv("./../../data/titanic/train.csv")
    test_raw = pd.read_csv("./../../data/titanic/test.csv")

    preprocessing = Preprocessing(train_raw, test_raw)
    preprocessing.run()

    model = LogisticRegression(preprocessing)
    model.fit("Survived ~ Pclass + Sex + Age + SibSp + Parch")
    model.predict()
    model.write_summary("./logistic_regression_summary.txt")
    model.write_predict("./logistic_regression_predict.csv")


if __name__ == '__main__':
    main()