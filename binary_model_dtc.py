import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import create_models

X_original, Y_original = create_models.get_data_for_original_model(mode='data_bin')
X_salinity, Y_salinity = create_models.get_data_for_original_model(mode='data_bin')
X_double, Y_double = create_models.get_data_for_original_model(mode='data_bin')

seed = 7
models = []
models.append(('DTC original data 0-1', DecisionTreeClassifier(), X_original, Y_original))
models.append(('DTC salinity data 0-1', DecisionTreeClassifier(), X_salinity, Y_salinity))
models.append(('DTC double data 0-1', DecisionTreeClassifier(), X_double, Y_double))
results = []
names = []

for name, model, X, Y in models:
    kfold = model_selection.KFold(n_splits=len(models), shuffle=False, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('{}: mean {} std({})'.format(name,
                                       cv_results.mean(),
                                       cv_results.std()))

fig = plt.figure('Binarno')
fig.suptitle('Rezultati DTC za klasificirane rezultate')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()