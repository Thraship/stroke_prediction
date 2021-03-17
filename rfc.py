# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:25:08 2021

@author: Amir Ostad
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def rfc(X, y):
    """
    Takes features database X and response data series y.
    Fits an optimized random forest classfier, plots confusion matrices,
    and reports different classification metrics.
    Returns the optimzied random forest classifier and training and
    test sets.
    """
    print(20 * "*" + " Random forest classifier modeling initiated!")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=23)
    # # with a single randomeforest:
    # rf = RandomForestClassifier()
    # print("rf = RandomForestClassifier() done!")
    # rf.fit(X_train, y_train)
    # y_train_hat = rf.predict(X_train)
    # y_test_hat = rf.predict(X_test)
    
    
    # with gridsearch optimization:
    rf = RandomForestClassifier(random_state=23)
    cv = cross_val_score(rf,X_train,y_train,cv=10)
    print("cross validation scores:\n", cv)
    print("the average of the cross validation scores: ", cv.mean())
    
    param_grid =  {'n_estimators': [10, 25, 50, 100, 200, 400, 800, 1000],
                                      'bootstrap': [True],
                                      'max_depth': [2, 5, 10, 15, 20],
                                      'max_features': ['auto','sqrt',10],
                                      'min_samples_leaf': [2,3,4,5,6],
                                      'min_samples_split': [2,3,4]}
    
    # param_grid =  {'n_estimators': [50],
    #                                   'bootstrap': [True],
    #                                   'max_depth': [2],
    #                                   'max_features': ['auto','sqrt',10],
    #                                   'min_samples_leaf': [2,3],
    #                                   'min_samples_split': [2,3]}
    
    cv = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = False,
                      n_jobs = -1)
    rf_clfs = cv.fit(X_train,y_train)
    
    # picking the best of them
    rf = rf_clfs.best_estimator_.fit(X_train,y_train)
    
    y_train_hat = rf.predict(X_train)
    y_test_hat = rf.predict(X_test)
    
    for matrix in [(confusion_matrix(y_test, y_test_hat),
                    "test set confusion matrix"),
                   (confusion_matrix(y_train, y_train_hat),
                    "training set confusion matrix")]:
        ax = plt.axes()
        sns.heatmap(matrix[0], annot=True, fmt='0.2f', ax=ax)
        ax.set_title(matrix[1])
        plt.show()
    
    # print("Here's the training set classification report:\n",
    #       classification_report(y_train, y_train_hat))
    # print("Here's the test set classification report:\n",
    #       classification_report(y_test, y_test_hat))
    
    return rf, X_train, X_test, y_train, y_test
