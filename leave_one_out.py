from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import create_models
import numpy


def create_nb(train_data_x, train_data_y, predict_data_x):
    gaussian_nb = GaussianNB()
    gaussian_prediction = gaussian_nb.fit(train_data_x, train_data_y).predict(predict_data_x)
    return  gaussian_prediction

# decision tree regression
def create_dt_regression(train_data_x, train_data_y, predict_data_x):
    regression_tree = DecisionTreeRegressor()
    regression_tree_prediction = regression_tree.fit(train_data_x, train_data_y).predict(predict_data_x)
    return  regression_tree_prediction


# decision tree classification
def create_dt_classifier(train_data_x, train_data_y, predict_data_x):
    classification_tree = DecisionTreeClassifier()
    classification_tree_prediction = classification_tree.fit(train_data_x, train_data_y).predict(predict_data_x)
    return  classification_tree_prediction


"""""""""""""""""""""
data with salinity
"""""""""""""""""""""


def get_data_with_salinity_model_LOO(mode):
    X, Y = create_models.get_data_with_salinity("data")
    X_bin, Y_bin = create_models.get_data_with_salinity("data_bin")
    result = []
    for i in range(len(X)):
        train_data_x = X.copy()
        predict_data_x = train_data_x.pop(i)
        train_data_y = Y.copy()
        train_data_y.pop(i)

        train_data_y_bin = Y_bin.copy()
        train_data_y_bin.pop(i)

        predict_data_x = numpy.array(predict_data_x).reshape(1, -1)
        if mode == "nb":
            result.append(create_nb(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtr":
            result.append(create_dt_regression(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtc":
            result.append(create_dt_classifier(train_data_x, train_data_y, predict_data_x))
        elif mode ==  "dtcb":
            result.append(create_dt_classifier(train_data_x, train_data_y_bin, predict_data_x))
    return result

"""""""""""""""""""""
data with double time
"""""""""""""""""""""

def get_original_data_double_time_model_LOO(mode):
    X, Y = create_models.get_data_for_double_time("data")
    X_bin, Y_bin = create_models.get_data_for_double_time("data_bin")
    result = []

    for i in range(len(X)):
        train_data_x = X.copy()
        predict_data_x = train_data_x.pop(i)
        train_data_y = Y.copy()
        train_data_y.pop(i)
        predict_data_x = numpy.array(predict_data_x).reshape(1, -1)

        train_data_y_bin = Y_bin.copy()
        train_data_y_bin.pop(i)

        if mode == "nb":
            result.append(create_nb(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtr":
            result.append(create_dt_regression(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtc":
            result.append(create_dt_classifier(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtcb":
            result.append(create_dt_classifier(train_data_x, train_data_y_bin, predict_data_x))
    return result

"""
    Original data
"""

def get_original_data_model_LOO(mode):
    X, Y = create_models.get_data_for_original_model('data')
    X_bin, Y_bin = create_models.get_data_for_original_model("data_bin")
    result = []
    for i in range(len(X)):
        train_data_x = X.copy()
        predict_data_x = train_data_x.pop(i)
        train_data_y = Y.copy()
        train_data_y.pop(i)
        predict_data_x = numpy.array(predict_data_x).reshape(1, -1)

        train_data_y_bin = Y_bin.copy()
        train_data_y_bin.pop(i)

        if mode == "nb":
            result.append(create_nb(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtr":
            result.append(create_dt_regression(train_data_x, train_data_y, predict_data_x))
        elif mode == "dtc":
            result.append(create_dt_classifier(train_data_x, train_data_y, predict_data_x))
        elif mode ==  "dtcb":
            result.append(create_dt_classifier(train_data_x, train_data_y_bin, predict_data_x))
    return result