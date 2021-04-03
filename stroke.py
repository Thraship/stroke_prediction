# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:33:31 2021

@author: Amir Ostad

In this model the objective is to predict the likelihood of patients getting
strokes based on input features such as age, gender, bmi, ....
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import rfc
import visualize
import clean_data

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
# print(df.shape)
# print(df.info())
# from df.info we can see that bmi column has Null values. Here, the null
#values are filled with the average of the bmi column.

X = df

# drop the "id" column. There is no use for patient ids in this anlysis.
X.drop("id", axis=1, inplace=True)

# print(X.gender.value_counts()) # checking different values for gender
# Only 1 instance with gender=Other. The rest are either male or female
# Removing the instance with gender=Other
X = X[X.gender != "Other"]
# print(X.gender.value_counts()) # rechecking gender values

visualize.visualize(X)

# extracting the response variable
y = X.pop("stroke")

X = clean_data.clean_data(X)
print(20 * "*" + " Data cleaning ended successfully!")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=23)
rf = rfc.rfc(X_train, X_test, y_train, y_test)
print(20 * "*" + " Machine learning modeling ended successfully!")

# save model using joblib
FILENAME = "saved_model.sav"
joblib.dump(rf, FILENAME)

# load the model form disk
loaded_model = joblib.load(FILENAME)
print("The optimized model: \n", loaded_model)
print("Model's accuracy = ", loaded_model.score(X_test, y_test))

# Which features are more important
feat_importances = pd.Series(loaded_model.feature_importances_,
                             index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh', color='teal')


