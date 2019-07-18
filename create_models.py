import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def get_data_with_salinity(mode):
    """
        Extracting data  for model that includes salinity

        mode:
            'nb' for naive bayes result
            'dtc' for decision tree classifier
            'dtr' for decision tree regressor
            'data' for returning dataset
            'result' for result provided in excel
            'result_bin' for result provided in excel as classes
            'dtc_bin' for results as classes
    """
    data_with_salinity = []
    salinity_bin_result = []
    salinity_result = []

    data = pd.read_excel(io='All_data.xlsx', sheet_name='Sheet1')

    for i in data.index:
        data_with_salinity.append([data['Suncevo zracenje'][i], data['Padaline 72h'][i], data['Salinitet'][i]])
        salinity_bin_result.append(data['Zagadenost binarno'][i])
        salinity_result.append(data['Zagadenost'][i])

    X = data_with_salinity[:int(2*len(data_with_salinity)/3)]
    Y = salinity_result[:int(2*len(data_with_salinity)/3)]
    Y_bin = salinity_bin_result[:int(2*len(data_with_salinity)/3)]
    predict_data = data_with_salinity[2*int(len(data_with_salinity)/3)+1:]
    result_data = salinity_result[2*int(len(data_with_salinity)/3)+1:]
    result_data_bin = salinity_bin_result[2*int(len(data_with_salinity)/3)+1:]

    # if only data is wanted, do not run model creation
    if mode == 'data':
        return data_with_salinity, salinity_result
    elif mode == 'data_bin':
        return data_with_salinity, salinity_bin_result
    elif mode == 'result':
        return result_data
    elif mode == 'result_bin':
        return result_data_bin

    # naive bayes
    if mode == 'nb':
        gaussian_nb_salinity = GaussianNB()
        gaussian_prediction_bez = gaussian_nb_salinity.fit(X, Y).predict(predict_data)
        return gaussian_prediction_bez

    # decision tree regression

    if mode == 'dtr':
        regression_tree = DecisionTreeRegressor()
        regression_tree_prediction_bez = regression_tree.fit(X, Y).predict(predict_data)
        return regression_tree_prediction_bez

    # decision tree classification
    if mode == 'dtc':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction_bez = classification_tree.fit(X, Y).predict(predict_data)
        return classification_tree_prediction_bez

    if mode == 'dtc_bin':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction_bez_bin = classification_tree.fit(X, Y_bin).predict(predict_data)
        return classification_tree_prediction_bez_bin


def get_data_for_double_time(mode):
    """
        Extracting data for model that takes both measuring from 10AM and 2PM

        mode:
            'nb' for naive bayes result
            'dtc' for decision tree classifier
            'dtr' for decision tree regressor
            'data' for returning dataset
            'result' for result provided in excel
            'result_bin' for result provided in excel as classes
            'dtc_bin' for results as classes
    """
    data = pd.read_excel(io='Original_data.xlsx', sheet_name='podaci')
    raw_data = []
    for i in data.index:
        if data['TOCKA'][i] == '3M' and data['VRIJEME'][i] in [10, 14]:
            raw_data.append(
                {
                    'date': data['DATUM'][i],
                    'time': data['VRIJEME'][i],
                    'rain': data['KPad72'][i],
                    'sun':  data['Sunce'][i],
                    'zag': data['EC'][i]
                }
            )

    double_time = []
    double_time_result = []
    double_time_result_bin = []
    for num, data in enumerate(raw_data):
        if num+1 != len(raw_data) and raw_data[num]['date'] == raw_data[num+1]['date']:
            double_time.append([raw_data[num]['sun'], raw_data[num+1]['sun'], raw_data[num]['rain'], raw_data[num+1]['rain']])
            double_time_result.append(raw_data[num+1]['zag'])
            double_time_result_bin.append(
                0 if raw_data[num+1]['zag'] < 300 else 1)

    X = double_time[:int(2*len(double_time)/3)]
    Y = double_time_result[:int(2*len(double_time)/3)]
    Y_bin = double_time_result_bin[:int(2*len(double_time)/3)]
    predict_data = double_time[2*int(len(double_time)/3)+1:]
    result_data = double_time_result[2*int(len(double_time)/3)+1:]
    result_data_bin = double_time_result_bin[2*int(len(double_time)/3)+1:]

    if mode == 'data':
        return double_time, double_time_result
    elif mode == 'data_bin':
        return double_time, double_time_result_bin
    elif mode == 'result':
        return result_data
    elif mode == 'result_bin':
        return result_data_bin

    # naive bayes

    if mode == 'nb':
        gaussian_nb_salinity = GaussianNB()
        gaussian_prediction = gaussian_nb_salinity.fit(X, Y).predict(predict_data)
        return gaussian_prediction

    # decision tree regression
    if mode == 'dtr':
        regression_tree = DecisionTreeRegressor()
        regression_tree_prediction = regression_tree.fit(X, Y).predict(predict_data)
        return regression_tree_prediction

    # decision tree classification
    if mode == 'dtc':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction = classification_tree.fit(X, Y).predict(predict_data)
        return classification_tree_prediction

    if mode == 'dtc_bin':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction_bin = classification_tree.fit(X, Y_bin).predict(predict_data)
        return classification_tree_prediction_bin


def get_data_for_original_model(mode):
    """
        Data used for original model

        mode:
            'nb' for naive bayes result
            'dtc' for decision tree classifier
            'dtr' for decision tree regressor
            'data' for returning dataset
            'result' for result provided in excel
            'result_bin' for result provided in excel as classes
            'dtc_bin' for results as classes
    """

    data = pd.read_excel(io='All_data.xlsx', sheet_name='Sheet2')
    original_data = []
    original_data_result = []
    original_data_result_bin = []

    for i in data.index:
        original_data.append([data['sunce'][i], data['padaline'][i]])
        original_data_result.append(data['zag'][i])
        original_data_result_bin.append(0 if data['zag'][i] < 300 else 1)

    X = original_data[:int(2*len(original_data)/3)]
    Y = original_data_result[:int(2*len(original_data)/3)]
    Y_bin = original_data_result_bin[:int(2*len(original_data)/3)]
    predict_data = original_data[2*int(len(original_data)/3)+1:]
    original_result = original_data_result[2*int(len(original_data)/3)+1:]
    original_result_bin = original_data_result_bin[2*int(len(original_data)/3)+1:]

    if mode == 'data':
        return original_data, original_data_result
    elif mode == 'data_bin':
        return original_data, original_data_result_bin
    elif mode == 'result':
        return original_result
    elif mode == 'result_bin':
        return original_result_bin

    # naive bayes

    if mode == 'nb':
        gaussian_nb_salinity = GaussianNB()
        gaussian_prediction_bez = gaussian_nb_salinity.fit(X, Y).predict(predict_data)
        return gaussian_prediction_bez

    # decision tree regression
    if mode == 'dtr':
        regression_tree = DecisionTreeRegressor()
        regression_tree_prediction_bez = regression_tree.fit(X, Y).predict(predict_data)
        return regression_tree_prediction_bez

    # decision tree classification
    if mode == 'dtc':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction_bez = classification_tree.fit(X, Y).predict(predict_data)
        return classification_tree_prediction_bez

    if mode == 'dtc_bin':
        classification_tree = DecisionTreeClassifier()
        classification_tree_prediction_bez_bin = classification_tree.fit(X, Y_bin).predict(predict_data)
        return classification_tree_prediction_bez_bin


if __name__ == '__main__':
    get_data_for_double_time(mode='data')
    get_data_for_original_model(mode='data')
    get_data_with_salinity(mode='data')
