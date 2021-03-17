# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:41:06 2021

@author: Amir Ostad
"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

def visualize(X):
    """
    plots histogram for numerical variables and bar charts to show counts for
    categorical variables of the dataframe X.
    """
    # data visualization
    # sns.pairplot(df) # you can try pairplot too
    X.hist(bins=20) # distributions of numerical values
    plt.show()
    """ A few things regarding the histograms:
        
        1- Normal blood sugar levels are less than 100 mg/dL after
        not eating (fasting) for at least eight hours. And they're less
        than 140 mg/dL two hours after eating. During the day, levels tend
        to be at their lowest just before meals. For most people without
        diabetes, blood sugar levels before meals hover around 70 to 80 mg/dL. 
        For some people, 60 is normal; for others, 90 is the norm.
        Thus, the average_glucose_level histogram makes sense.
        
        2- Age histogram nearly shows a normal distribution indicating proper
        sampling of people in different age groups.
        
        3- Hypertension, bmi, and stroke distributions indicate that
        these variables are binary.
        
        4- BMI is used to broadly define different weight groups in adults
        20 years old or older. The same groups apply to both men and women.
        Underweight: BMI is less than 18.5
        Normal weight: BMI is 18.5 to 24.9
        Overweight: BMI is 25 to 29.9
        Obese: BMI is 30 or more
        Thus the bmi histogram seems reasonable.
        
    """
    # data visualization: correlation heatmap
    ax = plt.axes()
    sns.heatmap(X.corr(), annot=True, fmt='0.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.show()
    """ A few things regarding the correlation heatmap:
        
        1- Stroke has the  highest correlation with age.
    
    """
    #data visualization: categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        sns.countplot(x=col, data=X, hue=col)
        plt.show()
    """Insight from the categorical plots:
        1- gender: there are enough instances of both males and females.
        2- ever_married: there are plenty of both married and 
        unmarried instances.
        3- work_type: there are different instances of folks with different 
                        types of occupations.
        4- Residence_type: good proportion of Urban & Rural instances 
        are present.
        5- smoking_status: there are plenty instances with different 
                            smoking statuses.
        """