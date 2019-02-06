import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Variables
dataFrame = pandas.read_csv("dataset1.csv", delimiter=",", header=None,
                            usecols=[2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24])

dataset = dataFrame.values

X = dataset[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]]
Y = dataset[:, 7]


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    model.summary()
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kFold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kFold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# evaluate model with standardized dataset
np.random.seed(seed)
estimators = [('standardize', StandardScaler()),
              ('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0))]
pipeline = Pipeline(estimators)
kFold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kFold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# define the model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


np.random.seed(seed)
estimators = [('standardize', StandardScaler()),
              ('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0))]
pipeline = Pipeline(estimators)
kFold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kFold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


np.random.seed(seed)
estimators = [('standardize', StandardScaler()),
              ('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0))]
pipeline = Pipeline(estimators)
kFold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kFold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
