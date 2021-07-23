from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

train_data = pd.read_csv('dataset/boston_housing_prices/train.csv')

test_data = pd.read_csv('dataset/boston_housing_prices/test.csv')

train_data.head()

print(train_data.head())