###################################################################
#  Model_2_Prcpt_LogReg.py
#
#  Wendy Slattery
#  Course: Artificial Intelligence
#  12/2/20
#  Final Project: Decision Tree and Perceptron Logical Regression
#  applied to Breast Cancer data set from
#  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
####################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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
def if_irradiated(i):
    if i == 'yes':
        return 1
    else:
        return 0


df['Irradiated'] = df['irradiate'].map(lambda i: if_irradiated(i))
# print(df['Irradiated'])

###################################################################################
#
# Model 2: Perceptron and Logistical Regression
#
###################################################################################

# try to set the graph to nearly separable data

# Action
ActionDf = df['Recurred'].map({0: "Cancer_Free", 1: "Recurred"}).astype(str)
# print(ActionDf)

# dropping all columns except Age_Range, Degree_Malignant, Tumor_Size
data = df.drop(columns=['recur_event', 'Recurred', 'age', 'menopause', 'Menopause', 'tumor_size', 'inv_nodes',
              'node_caps', 'deg_malig', 'breast', 'irradiate', 'breast_quad', 'Tumor_Size'])
print(data)


# train test split.
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.45, stratify=ActionDf, random_state=1)

# Display Training data
Y_traindf = Y_train.to_frame()
# print(Y_traindf)
Train = pd.concat([X_train, Y_traindf], axis=1, join='inner')
# print(Train)
TrainRecurred = Train.query('Recurred=="Recurred"')
TrainCancerFree = Train.query('Recurred=="Cancer_Free"')

ax = TrainRecurred.plot.scatter(x='Age_Range', y='Degree_Malignant', c='red', s=70, alpha=.07)
TrainCancerFree.plot.scatter(x='Age_Range', y='Degree_Malignant', c='green', ax=ax, s=70, alpha=.07)
plt.title('Training set age range vs degree malignant')
plt.xlabel('age range')
plt.ylabel('degree malignant')
plt.xticks(np.arange(0, 100, 10.0))
plt.yticks(np.arange(0, 3, 1.0))
plt.show()

# Display Testing data
Y_testdf = Y_test.to_frame()
# print(Y_traindf)
Test = pd.concat([X_test, Y_testdf], axis=1, join='inner')
# print(Train)
TestRecurred = Test.query('Recurred=="Recurred"')
TestCancerFree = Test.query('Recurred=="Cancer_Free"')

ax = TestRecurred.plot.scatter(x='Age_Range', y='Degree_Malignant', c='red', s=70, alpha=.07)
TestCancerFree.plot.scatter(x='Age_Range', y='Degree_Malignant', c='green', ax=ax, s=70, alpha=.07)
plt.title('Testing set age range vs degree malignant')
plt.xlabel('age range')
plt.ylabel('degree malignant')
plt.xticks(np.arange(0, 100, 10.0))
plt.yticks(np.arange(0, 3, 1.0))
plt.show()


#########################
# Logistic Regression
#########################


# Perceptron
print("Perceptron")
perceptron = lm.Perceptron(verbose=1)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)
print("\n\nPerceptron")
print("\tNumber of Features...", perceptron.n_features_in_)
print("\tColumns", X_train.columns)
print("\tCoefficients", perceptron.coef_)
print("\tIntercept", perceptron.intercept_)

print('\nAccuracy of perceptron on test set: {:.2f}'.format(perceptron.score(X_test, Y_test)))

print("Confusion_Matrix...")
confusion_matrixP = confusion_matrix(Y_test, Y_pred)
print(confusion_matrixP)
###########################
print("\n###############################")
# print("Logistic Regression")
logreg = LogisticRegression(verbose=0)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
print("\nLogistic Regression")
print("\tColumns", X_train.columns)
print("\tCoefficients", logreg.coef_)
print("\tIntercept", logreg.intercept_)

print('\nAccuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

print("Confusion_Matrix...")
confusion_matrixLG = confusion_matrix(Y_test, Y_pred)
print(confusion_matrixLG)
###########################
