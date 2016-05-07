#%matplotlib inline

# Machine Learning for Public Policy
# Assignment 2: Building an initial pipeline
# Name: Vi Nguyen

import matplotlib.pylab as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from sklearn import (preprocessing, cross_validation, svm, metrics, tree, 
    decomposition, svm)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import (LogisticRegression, Perceptron, 
    SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression)
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from readto_pd_df import *
from impute import *

'''
Following code has been minimally adapted from Rayid Ghani's magicloops
https://github.com/rayidghani/magicloops.git

'''
def define_clfs_params():

    clfs = {'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),\
    'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'), \
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), algorithm = "SAMME", n_estimators = 200),\
    'LR': LogisticRegression(penalty = 'l1', C = 1e5),\
    'SVM': svm.LinearSVC(),\
    'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),\
    'NB': GaussianNB(),\
    'DT': DecisionTreeClassifier(),\
    'SGD': SGDClassifier(loss = "hinge", penalty = "l2"),\
    'KNN': KNeighborsClassifier(n_neighbors = 3)}

    grid = {'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100],'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},\
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},\
    'SGD': { 'loss': ['hinge','log','perceptron'],\
    'penalty': ['l2','l1','elasticnet']},\
    'ET': { 'n_estimators': [1,10,100,1000,10000], \
    'criterion' : ['gini', 'entropy'],\
    'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],\
    'min_samples_split': [2,5,10]},\
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], \
    'n_estimators': [1,10,100,1000,10000]},\
    'GB': {'n_estimators': [1,10,100,1000,10000], \
    'learning_rate' : [0.001,0.01,0.05,0.1,0.5],\
    'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},\
    'NB' : {},\
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], \
    'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},\
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10]}, \
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],\
    'weights': ['uniform','distance'],\
    'algorithm': ['auto','ball_tree','kd_tree']}}
    return clfs, grid

def magic_loop(models_to_run, clfs, params, df, y_label):

    for n in range(1, 2):
        train, test = train_test_split(df, test_size = 0.2, random_state = 0)
        #print(train.dtypes)
        #print(test.dtypes)
        y_train = train[y_label].as_matrix()
        #print(y_train.dtypes)
        y_test = test[y_label].as_matrix()
        X_train = train.drop(y_label, axis = 1).as_matrix()
        X_test = test.drop(y_label, axis = 1).as_matrix()
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            model = models_to_run[index]
            print(model)
            parameter_values = params[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    print(clf)
                    #model = str(clf)
                    if model != 'SVM':
                        print('true')
                        k = .05
                        y_pred_probs = clf.predict_proba(X_test)[:,1]
                        print('precision, accuracy and f1 scores at k = {}: {}'.format(k, classification_metrics_at_k(y_test, y_pred_probs, k)))
                    all_y_pred = clf.predict(X_test)
                    #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                    #print threshold
                    #print('precision_recall_curve:')
                    #plot_precision_recall_n(y_test,y_pred_probs,clf)
                    print('auc curve: {}'.format(auc_metric(y_test, all_y_pred)))
                    print('MSE: {}'.format(metrics.mean_squared_error(y_test, all_y_pred)))
                    print()

                except IndexError as e:
                    print('Error:', e)
                    continue

'''

def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, 
    pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color = 'b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color = 'r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
'''

# Classification metrics
def classification_metrics_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.precision_score(y_true, y_pred, average = 'binary'),
        metrics.accuracy_score(y_true, y_pred),
        metrics.f1_score(y_true, y_pred, average = 'binary'))

def auc_metric(y_true, all_y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, all_y_pred, pos_label = 2)
    return metrics.auc(fpr, tpr)


def main(): 

    clfs, params = define_clfs_params()
    #models_to_run = ['RF', 'LR', 'SVM', 'NB', 'DT', 'KNN']
    #models_to_run = [ 'LR', 'SVM', 'NB', 'DT', 'KNN']
    models_to_run = [ 'LR', 'NB', 'DT', 'KNN', 'SVM']
    df = read('cs-training.csv', 'csv')
    df = fillna_mean(df)
    magic_loop(models_to_run, clfs, params, df, 'SeriousDlqin2yrs')

if __name__ == '__main__':
    main()