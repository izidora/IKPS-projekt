import matplotlib.pyplot as plt
import create_models
import leave_one_out

"""
    Double time data
"""

original_result_double_time = create_models.get_data_for_double_time(mode='result')
nb_result_double_time = create_models.get_data_for_double_time(mode='nb')
dtc_result_double_time = create_models.get_data_for_double_time(mode='dtc')
dtr_result_double_time = create_models.get_data_for_double_time(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - double time')
plt.plot(original_result_double_time, color='green', label='original')
plt.plot(nb_result_double_time, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-double_time')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - double time')
plt.plot(original_result_double_time, color='green', label='original')
plt.plot(dtc_result_double_time, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-double_time')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - double time')
plt.plot(original_result_double_time, color='green', label='original')
plt.plot(dtr_result_double_time, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-double_time')

"""
    Original data
"""

original_result = create_models.get_data_for_original_model(mode='result')
nb_result_original = create_models.get_data_for_original_model(mode='nb')
dtc_result_original = create_models.get_data_for_original_model(mode='dtc')
dtr_result_original = create_models.get_data_for_original_model(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - original data')
plt.plot(original_result, color='green', label='original')
plt.plot(nb_result_original, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-original_data')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - original data')
plt.plot(original_result, color='green', label='original')
plt.plot(dtc_result_original, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-original_data')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - original data')
plt.plot(original_result, color='green', label='original')
plt.plot(dtr_result_original, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-original_data')


"""
    salinity data
"""

original_result_salinity = create_models.get_data_with_salinity(mode='result')
nb_result_salinity = create_models.get_data_with_salinity(mode='nb')
dtc_result_salinity = create_models.get_data_with_salinity(mode='dtc')
dtr_result_salinity = create_models.get_data_with_salinity(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - salinity data')
plt.plot(original_result_salinity, color='green', label='original')
plt.plot(nb_result_salinity, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-salinity_data')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - salinity data')
plt.plot(original_result_salinity, color='green', label='original')
plt.plot(dtc_result_salinity, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-salinity_data')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - salinity data')
plt.plot(original_result_salinity, color='green', label='original')
plt.plot(dtr_result_salinity, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-salinity_data')

"""
    Decision tree classifier binarno
"""

original_result_salinity_bin = create_models.get_data_with_salinity(mode='result_bin')
dtc_result_salinity_bin = create_models.get_data_with_salinity(mode='dtc_bin')

original_result_original_data_bin = create_models.get_data_for_original_model(mode='result_bin')
dtc_result_original_bin = create_models.get_data_for_original_model(mode='dtc_bin')

original_result_double_data_bin = create_models.get_data_for_double_time(mode='result_bin')
dtc_result_double_data_bin = create_models.get_data_for_double_time(mode='dtc_bin')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - salinity data 0 - 1')
plt.plot(original_result_salinity_bin, color='green', label='original')
plt.plot(dtc_result_salinity_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-salinity_data_0-1')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - original data 0 - 1')
plt.plot(original_result_original_data_bin, color='green', label='original')
plt.plot(dtc_result_original_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-original_data_0-1')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - double data 0 - 1')
plt.plot(original_result_double_data_bin, color='green', label='original')
plt.plot(dtc_result_double_data_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-double_data_0-1')

"""
    Original data - LOO
"""

x, y = original_result_double_time = create_models.get_data_for_original_model(mode="data")
nb_result_original_LOO = leave_one_out.get_original_data_model_LOO(mode='nb')
dtc_result_original_LOO = leave_one_out.get_original_data_model_LOO(mode='dtc')
dtr_result_original_LOO = leave_one_out.get_original_data_model_LOO(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - original data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(nb_result_original_LOO, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-original_data-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - original data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtc_result_original_LOO, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-original_data-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - original data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtr_result_original_LOO, color='skyblue', label='Decision tree reg')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-original_data-LOO')

"""
    Salinity data - LOO
"""

x, y = create_models.get_data_with_salinity(mode="data")
nb_result_salinity_LOO = leave_one_out.get_data_with_salinity_model_LOO(mode='nb')
dtc_result_salinity_LOO = leave_one_out.get_data_with_salinity_model_LOO(mode='dtc')
dtr_result_salinity_LOO = leave_one_out.get_data_with_salinity_model_LOO(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - salinity data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(nb_result_salinity_LOO, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-salinity_data-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - salinity data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtc_result_salinity_LOO, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-salinity_data-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - salinity data - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtr_result_salinity_LOO, color='skyblue', label='Decision tree reg')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-salinity_data-LOO')

"""
    Double time - LOO
"""

x, y = create_models.get_data_for_double_time(mode="data")
nb_result_double_time_LOO = leave_one_out.get_original_data_double_time_model_LOO(mode='nb')
dtc_result_double_time_LOO = leave_one_out.get_original_data_double_time_model_LOO(mode='dtc')
dtr_result_double_time_LOO = leave_one_out.get_original_data_double_time_model_LOO(mode='dtr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Naive Bayes - double time - LOO')
plt.plot(y, color='green', label='original')
plt.plot(nb_result_double_time_LOO, color='skyblue', label='naive bayes')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Naive_Bayes-double_time-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - double time - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtc_result_double_time_LOO, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-double_time-LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree regression - double time - LOO')
plt.plot(y, color='green', label='original')
plt.plot(dtr_result_double_time_LOO, color='skyblue', label='Decision tree reg')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_regression-double_time-LOO')


"""
    Decision tree classifier binarno - LOO
"""

x, original_result_salinity_bin = create_models.get_data_with_salinity(mode='data_bin')
dtc_result_salinity_bin = leave_one_out.get_data_with_salinity_model_LOO(mode='dtcb')

x, original_result_original_data_bin = create_models.get_data_for_original_model(mode='data_bin')
dtc_result_original_bin = leave_one_out.get_original_data_model_LOO(mode='dtcb')

x, original_result_double_data_bin = create_models.get_data_for_double_time(mode='data_bin')
dtc_result_double_data_bin = leave_one_out.get_original_data_double_time_model_LOO(mode='dtcr')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - salinity data 0 - 1 - LOO')
plt.plot(original_result_salinity_bin, color='green', label='original')
plt.plot(dtc_result_salinity_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-salinity_data_0-1_LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - original data 0 - 1 - LOO')
plt.plot(original_result_original_data_bin, color='green', label='original')
plt.plot(dtc_result_original_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-original_data_0-1_LOO')

fig = plt.figure()
fig.suptitle('Usporedba mjerenja s Decision tree classification - double data 0 - 1 - LOO')
plt.plot(original_result_double_data_bin, color='green', label='original')
plt.plot(dtc_result_double_data_bin, color='skyblue', label='Decision tree class')
plt.legend()
plt.show()
fig.savefig('Usporedba_mjerenja_s_Decision_tree_classification-double_data_0-1_LOO')

"""
plt.figure('Original data-bez suhih')
plt.plot(original_result_double_time_bez)
plt.show()

plt.figure('Original data comparison prediction-suhi')
plt.plot(original_result_double_time, color='green', label='original')
plt.plot(nb_result_double_time, color='skyblue', label='naive bayes')
plt.plot(dtc_result_double_time, color='red', label='dt class')
plt.plot(dtr_result_double_time, color='yellow', label='dt reg')
plt.legend()
plt.show()

plt.figure('Original data-suhi')
plt.plot(original_result_double_time)
plt.show()

#sada za suhe i bezsuhih sa slanocom

original_result_slanoca_bez,original_result_slanoca = create_models.get_data_with_salinity(mode='result')
nb_result_slanoca_bez,nb_result_slanoca = create_models.get_data_with_salinity(mode='nb')
dtc_result_slanoca_bez,dtc_result_slanoca = create_models.get_data_with_salinity(mode='dtc')
dtr_result_slanoca_bez,dtr_result_slanoca = create_models.get_data_with_salinity(mode='dtr')

plt.figure('Prediction with salinity-bez suhih')
plt.plot(original_result_slanoca_bez, color='green', label='original')
plt.plot(nb_result_slanoca_bez, color='skyblue', label='naive bayes')
plt.plot(dtc_result_slanoca_bez, color='red', label='dt class')
plt.plot(dtr_result_slanoca_bez, color='yellow', label='dt reg')
plt.legend()
plt.show()

plt.figure('Original with salinity-bez suhih')
plt.plot(original_result_slanoca_bez)
plt.xlim(0,len(original_result_slanoca_bez)-1)
plt.show()

plt.figure('Prediction with salinity-suhi')
plt.plot(original_result_slanoca, color='green', label='original')
plt.plot(nb_result_slanoca, color='skyblue', label='naive bayes')
plt.plot(dtc_result_slanoca, color='red', label='dt class')
plt.plot(dtr_result_slanoca, color='yellow', label='dt reg')
plt.legend()
plt.show()

plt.figure('Original with salinity-suhi')
plt.plot(original_result_slanoca)
plt.show()
"""