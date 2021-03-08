###################################################################
#  Model_1_DT.py
#
#  Wendy Slattery
#  CAP 4601
#  12/2/20
#  Final Project: Decision Tree and Perceptron Logical Regression
#  applied to Breast Cancer data set from
#  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
###################################################################

import pandas as pd
from matplotlib import pyplot as plt
import pydotplus  # To create our Decision Tree Graph
from sklearn import tree  # For our Decision Tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from info_gain import info_gain


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
# Model 1: Decision Tree
#
###################################################################################

# convert these categorical variables into binary variables & drop rows missing data
data = pd.get_dummies(df[['Tumor_Size', 'Menopause', 'Age_Range', 'Degree_Malignant',
                          'inv_nodes', 'irradiate', 'breast_quad']])
data = data.dropna()
# print(data)

# Action attribute chosen is 'recur_event'
# Training the decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf_train = clf.fit(data, df['recur_event'])

# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)
                                # Gini decides which attribute/feature should be placed at the root node,
                                # which features will act as internal nodes or leaf nodes
# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create Decision Tree PDF
graph.write_pdf("DT1_Breast_Cancer.pdf")


######################################
# Run an information gain evaluation
######################################
print('\nInformation Gain on Recurrence')

ig = info_gain.info_gain(df['recur_event'], df['Tumor_Size'])
print('\tTumor Size=', ig)

ig = info_gain.info_gain(df['recur_event'], df['Menopause'])
print('\tMenopause=', ig)

ig = info_gain.info_gain(df['recur_event'], df['Age_Range'])
print('\tAge Range=', ig)

ig = info_gain.info_gain(df['recur_event'], df['Degree_Malignant'])
print('\tDegree Malignant=', ig)

ig = info_gain.info_gain(df['recur_event'], df['inv_nodes'])
print('\tNumber Involved Nodes=', ig)

ig = info_gain.info_gain(df['recur_event'], df['breast_quad'])
print('\tBreast Quadrant=', ig)

ig = info_gain.info_gain(df['recur_event'], df['irradiate'])
print('\tIrradiated=', ig)

print("\nHighest Information Gained from: "
      "\n\tDegree Malignant, Number Involved Nodes, Tumor Size, and Irradiated")


###################################################
# Decision Tree Based on Information Gained
# - using highest 4 ranked attributes above
###################################################

# convert these categorical variables into binary variables & drop rows missing data
data = pd.get_dummies(df[['Degree_Malignant', 'inv_nodes', 'Tumor_Size', 'irradiate']])
data = data.dropna()
# print(data)

# Action attribute chosen is 'recur_event'
# Training the decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf_train = clf.fit(data, df['recur_event'])

# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)
                                # Gini decides which attribute/feature should be placed at the root node,
                                # which features will act as internal nodes or leaf nodes
# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create Updated Decision Tree PDF
graph.write_pdf("DT2_Breast_Cancer.pdf")
print("\n* Decision Tree Created based on Information Gains.\n")


#########################################
# Determine Accuracy of Decision Tree
#########################################

# The decision tree classifier.
clf = tree.DecisionTreeClassifier()

# split data into test and training set
x = data  # Attributes we are evaluating
y = df['recur_event']  # What we are trying to predict

# this function randomly split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
# test_size=.3 means that our test set will be 30% of the train set.

# Training the Decision Tree
clf_train = clf.fit(x_train, y_train)

# Export/Print a decision tree in DOT format.
# print(tree.export_graphviz(clf_train, None))

# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)

# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create new decision tree PDF
graph.write_pdf("DT3_Breast_Cancer.pdf")


########################################################
# Graph Measuring Accuracy Based Upon Training Set Size
########################################################

# Accuracy
NumRuns = 5
TrainingSetSize = []
ScorePer = []
n = 0
for per in range(10, 85, 5):
    TrainingSetSize.append(per * .01)
    ScorePer.append(0)
    for i in range(NumRuns):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(per * .01), random_state=100)
        # test_size=.1 means that our test set will be 10% of the train set.

        # Training the Decision Tree
        clf_train = clf.fit(x_train, y_train)
        pred = clf_train.predict(x_test)  # parameter: new data to predict
        ScorePer[n] += accuracy_score(y_test, pred)
        print(ScorePer[n])
    ScorePer[n] /= NumRuns
    print(ScorePer[n])
    n += 1

# plot graph
d = pd.DataFrame({
    'accuracy': pd.Series(ScorePer),
    'training set size': pd.Series(TrainingSetSize)})

plt.plot('training set size', 'accuracy', data=d, label='accuracy')
plt.ylabel('accuracy')
plt.xlabel('training set size')
plt.show()


###################################################################################
# Redo updated decision tree - 5 different attributes based upon information gain
###################################################################################

# convert the categorical variables into binary variables
data = pd.get_dummies(df[['Degree_Malignant', 'inv_nodes', 'Tumor_Size', 'breast_quad', 'irradiate']])
data = data.dropna()
# print(data)

# split data into test and training set
x = data
y = df['recur_event']

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy")
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.45, random_state=100)

# Training the Decision Tree
# clf_train = clf.fit(x_train, y_train)

# Training the Decision Tree
clf_train = clf.fit(data, df['recur_event'])

pred = clf_train.predict(x_test)  # parameter: new data to predict
ScorePer = accuracy_score(y_test, pred)
print("Accuracy on Test Data =", ScorePer)

# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)

# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DT4_Breast_Cancer.pdf")


##############################################################
# Graph Measuring Accuracy Based Upon Depth of Decision Tree
##############################################################

max_depth = []
entropy = []
for i in range(1, 10):
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtree.fit(x_train, y_train)
    pred = dtree.predict(x_test)
    entropy.append(accuracy_score(y_test, pred))
    ####
    max_depth.append(i)

# plot graph
d = pd.DataFrame({
    'entropy': pd.Series(entropy),
    'max_depth': pd.Series(max_depth)})

plt.plot('max_depth', 'entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()


##############################################################
# No Depth Limit/all possible attributes
##############################################################

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy")
# Training the Decision Tree
clf_train = clf.fit(data, df['recur_event'])
# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)

# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DT5_Breast_Cancer_NoDpthLmt.pdf")


####################################################
#  With a Depth Limit Revealed Ideal by Graph Data
####################################################


# The decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
# Training the Decision Tree
clf_train = clf.fit(data, df['recur_event'])
# Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=list(clf_train.classes_), rounded=True, filled=True)

# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DT6_Breast_Cancer_DepthLimit.pdf")
