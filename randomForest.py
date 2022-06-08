# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from torch import rand

np.random.seed(10)
random.seed(10)

fpath = 'diabetes.csv'
data = pd.read_csv(fpath)
df = pd.DataFrame(data)

# remove rows with >= 2 zero values 
cols = ['BloodPressure', 'SkinThickness', 'BMI', 'Insulin', 'Glucose']
df_remove_zeros = df[~((df[cols] == 0).sum(1) >= 2)]

# calculate the median and replace the remaining 0 values with it
df_remove_zeros_med = df_remove_zeros.copy()
for col in cols:
    df_remove_zeros_med[col] = df_remove_zeros_med[col].replace(0, df_remove_zeros_med[df_remove_zeros_med[col] != 0][col].median())

# calculate the class mean and replace the 0 values with it for each feature
df_remove_zeros_mean = df_remove_zeros.copy()
for col in cols:
    pos_class_mean = df_remove_zeros_mean[(df_remove_zeros_mean[col] != 0) & (df_remove_zeros_mean['Outcome'] == 1)][col].mean()
    neg_class_mean = df_remove_zeros_mean[(df_remove_zeros_mean[col] != 0) & (df_remove_zeros_mean['Outcome'] == 0)][col].mean()
    df_remove_zeros_mean.loc[(df_remove_zeros_mean[col] == 0) & (df_remove_zeros_mean['Outcome'] == 1), col] = pos_class_mean
    df_remove_zeros_mean.loc[(df_remove_zeros_mean[col] == 0) & (df_remove_zeros_mean['Outcome'] == 0), col] = neg_class_mean

# replace 0s with median of column
df_median = df.copy()
for col in cols:
    df_median[col] = df_median[col].replace(0, df_median[df_median[col] != 0][col].median())

# replace 0s with mean of column - not the class mean
df_mean = df.copy()
for col in cols:
    df_mean[col] = df_mean[col].replace(0, df_mean[df_mean[col] != 0][col].mean())

# replace 0s with mean of column based on what class the row belongs to
df_class_mean = df.copy()
for col in cols:
    pos_class_mean = df_class_mean[(df_class_mean[col] != 0) & (df_class_mean['Outcome'] == 1)][col].mean()
    neg_class_mean = df_class_mean[(df_class_mean[col] != 0) & (df_class_mean['Outcome'] == 0)][col].mean()
    df_class_mean.loc[(df_class_mean[col] == 0) & (df_class_mean['Outcome'] == 1), col] = pos_class_mean
    df_class_mean.loc[(df_class_mean[col] == 0) & (df_class_mean['Outcome'] == 0), col] = neg_class_mean


# remove all rows that have an outlier, median replacement
df_cut = df_median.copy()
df_cut = df_cut[(np.abs(stats.zscore(df_cut)) < 3).all(axis=1)]

# remove outliers, class mean replacement
# ***this was the best performing pre-processing***
df_cut_class_mean = df_class_mean.copy()
df_cut_class_mean = df_cut_class_mean[(np.abs(stats.zscore(df_cut_class_mean)) < 3).all(axis=1)]


# flooring/capping outliers at 10th/90th percentile and median replacement
df_floorcap = df_median.copy()
for col in df_floorcap.columns[:-1]:
    percentile10 = df_floorcap[col].quantile(0.1)
    percentile90 = df_floorcap[col].quantile(0.9)

    df_floorcap[col] = np.where(df_floorcap[col] < percentile10, percentile10, df_floorcap[col])
    df_floorcap[col] = np.where(df_floorcap[col] > percentile90, percentile90, df_floorcap[col])

# flooring/capping outliers at 10th/90th percentile and mean replacement
df_mean_floorcap = df_mean.copy()

for col in df_mean_floorcap.columns[:-1]:
    percentile10 = df_mean_floorcap[col].quantile(0.1)
    percentile90 = df_mean_floorcap[col].quantile(0.9)

    df_mean_floorcap[col] = np.where(df_mean_floorcap[col] < percentile10, percentile10, df_mean_floorcap[col])
    df_mean_floorcap[col] = np.where(df_mean_floorcap[col] > percentile90, percentile90, df_mean_floorcap[col])

# normalize data
df_normed = df_median.copy()
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df_normed[cols_to_norm] = StandardScaler().fit_transform(df_normed[cols_to_norm])

"""
For the sake of clarity and conciseness, will only use the df_cut_class_mean dataframe from here on out. This was the best performing
method of cleaning the data and the process was the same for all other dataframes.
"""

# create X and y for data
X = df_cut_class_mean[df_cut_class_mean.columns[:-1]]
y = df_cut_class_mean[df_cut_class_mean.columns[-1]]

# oversample the data
sampler = RandomOverSampler(sampling_strategy='minority', random_state=10)
X, y = sampler.fit_resample(X,y)

# train-test-split with 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=10)


# initialize basic RF classifier with default parameters
# this will be used in the gridsearchCV model testing
# also provides a 'baseline' RF accuracy score 
basic_clf = RandomForestClassifier(random_state=10)
basic_clf.fit(X_train, y_train)
basic_score = basic_clf.score(X_test, y_test)

print('--- RF Model Accuracies ---')
print(f'Baseline RF accuracy score: {basic_score}')


# parameters and potential values to test in the gridsearchCV 
param_grid = { 
    'n_estimators': [90, 100, 125, 150],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [None,8,9,10,11,12],
    'criterion' : ['gini', 'entropy'],
    'class_weight' : ['balanced', 'balanced_subsample', None],
    'min_samples_split' : [2,3]
}

# gridsearch for best parameters using grid above
cv_rf = GridSearchCV(estimator=basic_clf, param_grid=param_grid, cv=5)
cv_rf.fit(X_train, y_train)

# create and fit RF model with the best parameters from the gridsearch
cv_best_params = RandomForestClassifier(**cv_rf.best_params_, random_state=10)
cv_best_params.fit(X_train, y_train)
cv_score = cv_best_params.score(X_test, y_test)

print(f'gridsearchCV best test score: {cv_score}')

print('--- Parameters ---')
print(f'Best parameters: {cv_rf.best_params_}')


"""
Generate the accuracies and confusion matrix
"""

# preds = cv_best_params.predict(X_test)
preds = basic_clf.predict(X_test)

accs = precision_recall_fscore_support(y_test, preds, average='binary')

print('--- Other Accuracies ---')
print(f'Precision: {accs[0]}')
print(f'Recall: {accs[1]}')
print(f'F1-Score: {accs[2]}')


conf_mat = confusion_matrix(y_test, preds, labels=cv_best_params.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=cv_best_params.classes_)

disp.plot()

plt.show()