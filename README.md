# Data Science, Stroke Prediction

## Project Overview
- The objective is to predict the likelihood of patients getting strokes based on input features such as age, gender, bmi, ...
- Dataset was attained from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
- Cleaned data from over 5000 patients
- Created and optimized a Random Forest Classification model by using GridSearchCV to reach 94.4% accuracy to predict stroke probability
- Model was saved using joblib library

## Code and resources used
- Python version: 3.8
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, joblib
- Dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## Data cleaning
Data had to be cleaned to be usable for the Random Forest Classification model. The following measures were taken:
- Numerical and categorical features were identified
- Null values for numerical features were filled with their corresponding mean values (bmi column was the only numerical features with null values)
- The gender feature has 1 instance with gender=Other. The rest are either male or female. The instance with gender=Other was removed
- Dummy variables were created for categorical features

## EDA
In this section, I looked at the distribution and value counts of the features. 
  1. Numerical features:
      - Normal blood sugar levels are less than 100 mg/dL after
        not eating (fasting) for at least eight hours. And they're less
        than 140 mg/dL two hours after eating. During the day, levels tend
        to be at their lowest just before meals. For most people without
        diabetes, blood sugar levels before meals hover around 70 to 80 mg/dL. 
        For some people, 60 is normal; for others, 90 is the norm.
        Thus, the average_glucose_level histogram makes sense.
      - Age histogram nearly shows a normal distribution indicating proper
        sampling of people in different age groups.
      - Hypertension, bmi, and stroke distributions indicate that
        these variables are binary.
      - BMI is used to broadly define different weight groups in adults
        20 years old or older. The same groups apply to both men and women.
        Underweight: BMI is less than 18.5
        Normal weight: BMI is 18.5 to 24.9
        Overweight: BMI is 25 to 29.9
        Obese: BMI is 30 or more. 
        Thus the bmi histogram seems reasonable.
        
Below you can see the distributions of numerical features plus the response variable, i.e., stroke:

![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/1-histograms.png "histograms")

Below is the correlation heatmap of the numerical features and response variable. As you can see, stroke has the  highest correlation with age.

![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/2-correlation_heatmap.png "correlation heatmap")


   2. Categorical features:
        - gender: there are enough instances of both males and females
        - ever_married: there are plenty of both married and unmarried instances
        - work_type: there are instances of folks with all different types of occupations
        - Residence_type: good proportion of Urban & Rural instances are present
        - smoking_status: there are plenty instances with different smoking statuses
        - 
Below are the value counts plots for categorical features:

![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/3-gender.png "gender value counts")
![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/4-married.png "married-unmarried value counts")
![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/5-worktype.png "worktype value counts")
![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/6-residencetype.png "residencetype value counts")
![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/7-smokestatus.png "smoke status value counts")


## Model Building

First, I cleaned the data, removed unwanted rows, filled null values, and created dummy variable for categorical features. Then, I split the data into training and test sets with a test size of 20%. 

Random Forest Classification was used for this prediction. The model was optimized by using GridSearchCV. The best fit accuracy was 94.4%.

## Model Performance

The best fit RandomForestClassifier had an accuracy of 94.4%.

Below you can see the confusion matrix for both the training and test sets as well.

![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/8-test_confusionmatrix.png "test confusion matrix")
![alt_text](https://github.com/Thraship/stroke_prediction/blob/master/plots/9-train_confusionmatrix.png "training confusion matrix")

