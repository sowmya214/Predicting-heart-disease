import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv("C:/Users/buddh/PycharmProjects/heart/heart.csv")

print("Number of instances: ", len(data))

#removing null and nan values

print("number of null and nan values in dataset:")
print("null: ", data.isnull().sum().sum())
print("nan: ", data.isna().sum().sum())

print("column names")
print(data.columns)

print("Original dataset:")
print("Percentage of disease: ", (len(data[data['target'] == 1]) / len(data)) * 100)
print("Percentage of no disease: ", (len(data[data['target'] == 0]) / len(data)) * 100)

X = data.drop('target', axis=1)
Y = data['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
print("\nPercentage of disease in train: ", (len(y_train[y_train == 1]) / len(y_train)) * 100)
print("Percentage of no disease in train : ", (len(y_train[y_train == 0]) / len(y_train)) * 100)

# correlation matrix

figure = plt.subplot()
training_data = x_train.join(y_train)
correlation_matrix = x_train.join(y_train).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm_r', annot_kws={'size': 20})
figure.set_title("Correlation Matrix", fontsize=14)
# plt.show()

f, axes = plt.subplots(ncols=5, nrows=3, figsize=(20, 4))
col = 0
row = 0
for column in training_data.columns:
    sns.boxplot(x='target', y=column, data=training_data, ax=axes[row][col])
    # axes[0][0].set_title(column, " vs target")
    if (col + 1) % 5 == 0:
        col = 0
        row += 1
    else:
        col += 1
# plt.show()

# based on boxplots, removing outliers for age, trestbps

remove_outliers_cols = ['age', 'trestbps']
for column in remove_outliers_cols:
    tmp = training_data[column].values
    q1, q3 = np.percentile(tmp, 25), np.percentile(tmp, 75)
    iqr = q3 - q1
    cut_off = iqr * 1.5
    lower_boundary, upper_boundary = q1 - cut_off, q3 + cut_off
    outliers = [x for x in tmp if x < lower_boundary or x > upper_boundary]
    training_data = training_data.drop(training_data[(training_data[column] > upper_boundary)].index)
    training_data = training_data.drop(training_data[(training_data[column] < lower_boundary)].index)
x_train = training_data.drop('target', axis=1)
y_train = training_data['target']

print()


# SVC

svc = SVC()
svc.fit(x_train, y_train)
training_score = cross_val_score(svc, x_train, y_train, cv=5)
print("Classifiers: ", svc.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100,
      "% accuracy score")

svc_params = {'C': [0.1, 0.5, 1, 10], 'kernel': ['rbf', 'poly', 'sigmoid']}
new_svc = GridSearchCV(SVC(), svc_params)
new_svc.fit(x_train, y_train)
opt_svc = new_svc.best_estimator_
svc_score = cross_val_score(opt_svc, x_train, y_train, cv=5)
print("Classifiers: ", opt_svc.__class__.__name__, "Has a training score of", round(svc_score.mean(), 2) * 100,
      "% accuracy score")
print()

# LINEAR DISCRIMINANT ANALYSIS

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
training_score = cross_val_score(lda, x_train, y_train, cv=5)
print("Classifiers: ", lda.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100,
      "% accuracy score")

lda_params = {'solver': ['svd', 'lsqr', 'eigen']}
new_lda = GridSearchCV(LinearDiscriminantAnalysis(), lda_params)
new_lda.fit(x_train, y_train)
opt_lda = new_lda.best_estimator_
lda_score = cross_val_score(opt_lda, x_train, y_train, cv=5)
print("Classifiers: ", opt_lda.__class__.__name__, "Has a training score of", round(lda_score.mean(), 2) * 100,
      "% accuracy score")
print()

# TESTING SCORES

score = svc.score(x_test, y_test) * 100
print("\nSupport Vector Classifier test score: ", score)
score = opt_svc.score(x_test, y_test) * 100
print("opt Support Vector Classifier test score: ", score, "\n")

score = lda.score(x_test, y_test) * 100
print("Linear Discriminant Analysis test score: ", score)
score = opt_lda.score(x_test, y_test) * 100
print("opt Linear Discriminant Analysis test score: ", score, "\n")

classifiers = [svc, opt_lda]
for clf in classifiers:
    metrics.plot_roc_curve(clf, x_test, y_test)
    # .figure_.suptitle("ROC curve comparison")

plt.show()
