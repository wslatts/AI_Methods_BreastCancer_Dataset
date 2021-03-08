###################################################################
#  Model_3_NBayes.py
#
#  Wendy Slattery
#  CAP 4601
#  12/2/20
#  Final Project: Decision Tree and Perceptron Logical Regression
#  applied to Breast Cancer data set from
#  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
####################################################################

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# 1) Data Set: Breast Cancer
###############################
# read breast-cancer.data file
# configure data for processing
###############################

file = r"breast-cancer.data"
df = pd.read_csv(file)
df.columns = ['recur_event', 'age', 'menopause', 'tumor_size', 'inv_nodes',
              'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiate']
# print(df)


#######################################################################
# Discrete Attributes: set up attributes and one target (recur_event)
#  age, tumor_size, deg_malig, irradiate, menopause, inv_nodes
#######################################################################

# Convert 'age' attribute to discrete ranges
def age_range(a):
    if a == '10-19':
        return 10
    elif a == '20-29':
        return 20
    elif a == '30-39':
        return 30
    elif a == '40-49':
        return 40
    elif a == '50-59':
        return 50
    elif a == '60-69':
        return 60
    elif a == '70-79':
        return 70
    elif a == '80-89':
        return 80
    elif a == '90-99':
        return 90
    else:
        return 'other age'


df['Age_Range'] = df['age'].map(lambda a: age_range(a))
# print(df['Age_Range'])


# Convert 'tumor_size' attribute to discrete ranges
def tumor_size_range(t):
    if t == '0-4':
        return 1
    elif t == '5-9':
        return 2
    elif t == '10-14':
        return 3
    elif t == '15-19':
        return 4
    elif t == '20-24':
        return 5
    elif t == '25-29':
        return 6
    elif t == '30-34':
        return 7
    elif t == '35-39':
        return 8
    elif t == '40-44':
        return 9
    elif t == '50-54':
        return 10
    elif t == '55-59':
        return 11
    else:
        return 0


df['Tumor_Size'] = df['tumor_size'].map(lambda t: tumor_size_range(t))
# print(df['Tumor_Size'])


# Convert 'deg_malig' attribute to discrete ranges
def deg_malig_range(dm):
    if dm == 1:
        return '1'
    elif dm == 2:
        return '2'
    elif dm == 3:
        return '3'
    else:
        return 0


df['Degree_Malignant'] = df['deg_malig'].map(lambda dm: deg_malig_range(dm))
# print(df['Degree Malignant'])


# Convert 'menopause' attribute to discrete ranges
def menopause_range(m):
    if m == 'lt40':
        return 3
    elif m == 'ge40':
        return 2
    elif m == 'premeno':
        return 1
    else:
        return 0


df['Menopause'] = df['menopause'].map(lambda m: menopause_range(m))
# print(df['Menopause'])


# Convert 'recur_event' target attribute to boolean
def recurred(r):
    if r == 'recurrence-events':
        return 1
    else:
        return 0


df['Recurred'] = df['recur_event'].map(lambda r: recurred(r))
# print(df['Recurred'])


# Convert 'irradiate' target attribute to boolean
def if_irradiated(ir):
    if ir == 'yes':
        return 1
    else:
        return 0


df['Irradiated'] = df['irradiate'].map(lambda ir: if_irradiated(ir))
# print(df['Irradiated'])

###################################################################################
#
# Model 3: Naive Bayes
#
###################################################################################

# Define two(X)features: Age_Range, Degree_Malignant with one Y classifier, recurred or cancer free
# goal: find sets that graph near separably
# The class(Y) data set/ Action
ActionDf = df['Recurred'].map({1: 'Recurred', 0: 'Cancer_Free'}).astype(str)
# print('ActionDF:\n', ActionDf)


"""
print("\nChecking Recurred Column for accurate data reflection:")
cancerFree = df.query('Recurred == 0')
recur = df.query('Recurred == 1')
print("Cancer Free:\n", cancerFree)
print("Recurred:\n", recur)
"""

# dropping all unused columns except Age_Range, Degree_Malignant, Irradiated
data = df.drop(columns=['recur_event', 'Recurred', 'age', 'menopause', 'Menopause', 'tumor_size', 'inv_nodes',
              'node_caps', 'deg_malig', 'breast', 'irradiate', 'breast_quad', 'Tumor_Size'])

# ensure data is in binary variable form
data = pd.get_dummies(data)
data.dropna()
print("\nData Example:\n", data)


#####################################################
# Part 1:
# A) Simple GaussianNB
#####################################################

# Train Test Split.
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.55, stratify=ActionDf, random_state=1)

model = GaussianNB()
model.fit(X_train, Y_train)

expected = Y_test
predicted = model.predict(X_test)
print("\nA: Simple Gaussian\nNaive Bayes")
print("-------------------")
print(" * Params", model.get_params())
print(" * Class labels", model.get_params())

print(" * Class labels", model.classes_)
print(" * Probability of each class", model.class_prior_)
print(" * Absolute additive value to variances", model.epsilon_)
print(" * Variance of each feature per class", model.sigma_)
print(" * Mean of each feature per class", model.theta_)


print('\nAccuracy on test set: {:.2f}'.format(model.score(X_test, Y_test)))
print("Confusion_Matrix...")
print(metrics.confusion_matrix(expected, predicted))

print("\n--> Part 1 Analysis: With no priors and no weights, the simple gaussian analysis\n"
      "\tresults in an accuracy of 70% and the highest achievable with this classifier.\n")


###########################################################################
# Part 2: Adding sample weights to one class over another
# Higher weights force the classifier to put more emphasis on these points.
# Sample GaussianNB weight
###########################################################################

# Train Test Split.
print("\n####################")
print("B: Weighted Gaussian\n")
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.55, stratify=ActionDf, random_state=1)

model = GaussianNB()

print("#B1 More Emphasis on Recurred than Cancer Free")
print("-------------------------------------------")
sample_weight = Y_train.map({'Cancer_Free': .40, 'Recurred': .60}).astype(float)


model.fit(X_train, Y_train, sample_weight)

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print(" * Params", model.get_params())

print(" * Class labels", model.classes_)
print(" * Probability of each class", model.class_prior_)
print(" * Absolute additive value to variances", model.epsilon_)
print(" * Variance of each feature per class", model.sigma_)
print(" * Mean of each feature per class", model.theta_)
print("\nAccuracy on test set: {:.2f}".format(model.score(X_test, Y_test)))
print("Confusion_Matrix...")
print(metrics.confusion_matrix(expected, predicted))

print("\n------------------->")
print("#B2 more emphasis on Cancer Free than Recurred")
print("-------------------------------------------")
sample_weight = Y_train.map({'Cancer_Free': .85, 'Recurred': .15}).astype(float)


model.fit(X_train, Y_train, sample_weight)

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print(" * Params", model.get_params())

print(" * Class labels", model.classes_)
print(" * Probability of each class", model.class_prior_)
print(" * Absolute additive value to variances", model.epsilon_)
print(" * Variance of each feature per class", model.sigma_)
print(" * Mean of each feature per class", model.theta_)
print("\nAccuracy on test set: {:.2f}".format(model.score(X_test, Y_test)))
print("Confusion_Matrix...")
print(metrics.confusion_matrix(expected, predicted))

print("\n--> Part 2 Analysis: With no priors, yet adding various sample weights to either recurred or\n"
      "\tcancer free. The accuracy seems to lean toward recurred and thus we need not add much more weight\n"
      "\tto recurred to get the higher 70% result, but we do need to add substantial more to cancer free\n"
      "\tto get the accuracy there to match\n")


######################################################
# C) Preset Priors GaussianNB
######################################################

print("\n#############################")
print("#C Split by GaussianNB Weight")
print("-----------------------------")

# Split train, test for calibration
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.55, stratify=ActionDf, random_state=1)

model = GaussianNB(priors=[.85, .15])

model.fit(X_train, Y_train)

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print(" *Params", model.get_params())

print(" * Class labels", model.classes_)
print(" * Probability of each class", model.class_prior_)
print(" * Absolute additive value to variances", model.epsilon_)
print(" * Variance of each feature per class", model.sigma_)
print(" * Mean of each feature per class", model.theta_)
print("\nAccuracy on test set: {:.2f}".format(model.score(X_test, Y_test)))
print("Confusion_Matrix...")
print(metrics.confusion_matrix(expected, predicted))

print("\n--> Setting Priors:  Here we have added priors.  By tweaking these values and using to my advantage the\n"
      "\tknowledge of the weighted values above, I emphasized priors to favor cancer free and was able to observe\n"
      "\tsimilar findings in the accuracy and confusion matrix for weighted classes above.\n")


###########################
# D) Smoothing GaussianNB
###########################

# var_smoothing float, default=1e-9
# Portion of the largest variance of all features that is added to variances for calculation stability.

print("\n####################")
# split train, test for calibration
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.72, stratify=ActionDf, random_state=1)

print("#D Var Smoothing")
print("--------------------")
for i in range(1, 100, 1):
    model = GaussianNB(var_smoothing=i/10000)
    model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
    expected = Y_test
    predicted = model.predict(X_test)
    print('var_smoothing=', i/10000, 'Accuracy on test set: {:.2f}'.format(model.score(X_test, Y_test)))
