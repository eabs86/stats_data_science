""" 
Origem: 
https://www.kaggle.com/code/rafjaa/resampling-strategies-for-imbalanced-datasets

"""

"""
Imbalanced datasets

In this kernel we will know some techniques to handle highly unbalanced datasets, 
with a focus on resampling. The Porto Seguro's Safe Driver Prediction competition, 
used in this kernel, is a classic problem of unbalanced classes, since insurance claims 
can be considered unusual cases when considering all clients. Other classic examples 
of unbalanced classes are the detection of financial fraud and attacks on 
computer networks.

"""

import pandas as pd
import numpy as np


df_train = pd.read_csv('../assets/porto_seguro/train.csv')

target_count = df_train.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')

""" 
The metric trap

One of the major issues that novice users fall into when dealing with unbalanced datasets
relates to the metrics used to evaluate their model. Using simpler metrics like 
accuracy_score can be misleading. In a dataset with highly unbalanced classes, 
if the classifier always "predicts" the most common class without performing any 
analysis of the features, it will still have a high accuracy rate, obviously illusory.

"""

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Remove 'id' and 'target' columns
labels = df_train.columns[2:]

X = df_train[labels]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Now let's run the same code, but using only one feature (which should drastically reduce the accuracy of the classifier):

model = XGBClassifier()
model.fit(X_train[['ps_calc_01']], y_train)
y_pred = model.predict(X_test[['ps_calc_01']])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

""" 
As we can see, the high accuracy rate was just an illusion. In this way, the choice of the metric
used in unbalanced datasets is extremely important. In this competition, the evaluation metric is 
the Normalized Gini Coefficient, a more robust metric for imbalanced datasets, that ranges from 
approximately 0 for random guessing, to approximately 0.5 for a perfect score.

"""

""" 
Confusion matrix

An interesting way to evaluate the results is by means of a confusion matrix, which shows the 
correct and incorrect predictions for each class. In the first row, the first column indicates 
how many classes 0 were predicted correctly, and the second column, how many classes 0 were 
predicted as 1. In the second row, we note that all class 1 entries were erroneously predicted 
as class 0.

Therefore, the higher the diagonal values of the confusion matrix the better, indicating many 
correct predictions.

"""

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

""" 
Resampling

A widely adopted technique for dealing with highly unbalanced datasets is called resampling. 
It consists of removing samples from the majority class (under-sampling) and / or adding more examples 
from the minority class (over-sampling).

Despite the advantage of balancing classes, these techniques also have their weaknesses (there is no 
free lunch). The simplest implementation of over-sampling is to duplicate random records from the 
minority class, which can cause overfitting. In under-sampling, the simplest technique involves 
removing random records from the majority class, which can cause loss of information.

Let's implement a basic example, which uses the DataFrame.sample method to get random samples 
each class:

"""

# Class count
count_class_0, count_class_1 = df_train.target.value_counts()

# Divide by class
df_class_0 = df_train[df_train['target'] == 0]
df_class_1 = df_train[df_train['target'] == 1]

# Random under-sampling

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)');

# Random over-sampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)')

""" 
Python imbalanced-learn module

A number of more sophisticated resapling techniques have been proposed in the scientific literature.

For example, we can cluster the records of the majority class, and do the under-sampling by removing
records from each cluster, thus seeking to preserve information. In over-sampling, instead of creating 
exact copies of the minority class records, we can introduce small variations into those copies, 
creating more diverse synthetic samples.

Let's apply some of these resampling techniques, using the Python library imbalanced-learn. 
It is compatible with scikit-learn and is part of scikit-learn-contrib projects.

"""

import imblearn

#For ease of visualization, let's create a small unbalanced sample dataset using the make_classification method:

from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

df = pd.DataFrame(X)
df['target'] = y
df.target.value_counts().plot(kind='bar', title='Count (target)');

#We will also create a 2-dimensional plot function, plot_2d_space, to see the data distribution:

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
"""
Because the dataset has many dimensions (features) and our graphs will be 2D,
we will reduce the size of the dataset using Principal Component Analysis (PCA):

"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

# Random under-sampling and over-sampling with imbalanced-learn

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X,y)
id_rus = rus.sample_indices_

print('Removed indexes:', id_rus)

plot_2d_space(X_rus, y_rus, 'Random under-sampling')

# Random over-sampling 

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)

print(X_ros.shape[0] - X.shape[0], 'new random picked points')

plot_2d_space(X_ros, y_ros, 'Random over-sampling')

""" 
Under-sampling: Tomek links

Tomek links are pairs of very close instances, but of opposite classes. 
Removing the instances of the majority class of each pair increases the space between the two classes, 
facilitating the classification process.

In the code below, we'll use ratio='majority' to resample the majority class.

"""

from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)
id_tl = tl.sample_indices_

print('Removed indexes:', id_tl)

plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')

""" 
Under-sampling: Cluster Centroids

This technique performs under-sampling by generating centroids based on clustering methods. 
The data will be previously grouped by similarity, in order to preserve information.

In this example we will pass the {0: 10} dict for the parameter ratio, to preserve 10 elements 
from the majority class (0), and all minority class (1) .

"""

from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids()
X_cc, y_cc = cc.fit_resample(X, y)

plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

""" 
Over-sampling: SMOTE

SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority 
class, based on those that already exist. It works randomly picingk a point from the minority class 
and computing the k-nearest neighbors for this point. The synthetic points are added between the 
chosen point and its neighbors.

"""

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X, y)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')


""" 
Over-sampling followed by under-sampling
Now, we will do a combination of over-sampling and under-sampling, 
using the SMOTE and Tomek links techniques:

"""

from imblearn.combine import SMOTETomek

smt = SMOTETomek()
X_smt, y_smt = smt.fit_resample(X, y)

plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')