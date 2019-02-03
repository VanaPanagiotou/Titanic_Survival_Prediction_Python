# Prediction of the survival of passengers in Titanic in Python language


## Table of Contents
* [Project Description](#Project-Description)
* [Libraries Required](#Libraries-Required)
* [Data](#Data)
* [Descriptive Statistics and Exploratory Analysis](#Descriptive-Statistics-and-Exploratory-Analysis)
* [Prediction](#Prediction)
* [Conclusion](#Conclusion)


## <a name="Project-Description"></a> Libraries Required

Titanic was a passenger liner that sank on 15 April 1912 during its maiden voyage from the UK to New York City after colliding with an iceberg. Only 722 out of 2224 passengers survived in this disaster. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
The goal of this project is to predict what sorts of people were likely to survive. For more information see [Titanic – Machine Learning from Disaster](https://www.kaggle.com/c/titanic) presented by [Kaggle](https://www.kaggle.com).



## <a name="Libraries-Required"></a> Packages Required

Here are the required libraries to run the code properly:
```
numpy
pandas
seaborn
matplotlib
os
sklearn
graphviz
multiprocessing
xgboost
keras
tensorflow
```


## <a name="Data"></a> Data


As with most Kaggle competitions, the Titanic data consists of a training set and a testing set, both of which are .csv files:
* The training set contains data for a subset of passengers including the outcomes or “ground truth” (i.e. the “Survived” response variable). We see that the training set has 891 observations (rows) and 12 variables. This set will be used to build the machine learning models. 

* The testing set contains data for a subset of passengers without providing the ground truth, since the project’s objective is to predict these outcomes. The testing set has 418 observations and 11 variables, since the “Survived” variable is missing. This set will be used to see how well the developed model performs on unseen data. 

A description of the variables that are encountered in the Titanic dataset is given in the next Table. 


| Variable Name | Variable Description | Possible Values | Categorical/Numerical |
| --- | --- | --- | --- |
| `PassengerId` | Observation Number | 1, 2, …, 1309 | Numerical |
| `Survived` | Survival | 1 = Yes, 0 = No | Categorical |
| `Pclass` | Passenger Class | 1 = 1st, 2 = 2nd, 3 = 3rd | Categorical |
| `Name` | Passenger Name | Braund, Mr. Owen Harris, etc. | Categorical |
| `Sex` | Sex of Passenger | male, female | Categorical |
| `Age` | Age of Passenger | 0.17 – 80 | Numerical |
| `SibSp` | No. of Siblings/Spouses Aboard | 0 – 8 | Numerical |
| `Parch` | No. of Parents/Children Aboard | 0 – 9 | Numerical |
| `Ticket` | Ticket Number | A/5 21171, 3101282, etc. | Categorical |
| `Fare` | Passenger Fare | 0 – 512.3292 | Numerical |
| `Cabin` | Passenger Cabin | C85, E46, etc. | Categorical |
| `Embarked` | Port of Embarkation | C = Cherbourg, <br/> Q = Queenstown, <br/> S = Southampton | Categorical |


## <a name="Descriptive-Statistics-and-Exploratory-Analysis"></a> Descriptive Statistics and Exploratory Analysis 
First, we are going to take a look at the data and examine their relationships. In addition, we have to find how many missing values we have and in which variables and replace them with sensible values.

We also use some visualizations in order to better understand the relationships between the variables.


## <a name="Prediction"></a> Prediction
Now, we will build and fit some models to the training set and we will compute their accuracy on the testing set. For this purpose, we will build seven different models: **Decision Tree (DT)**, **Random Forest (RF)**, **Extra-Tree**, **Logistic Regression (LR)**, **Gradient Boosting Machine (GBM)**, **XGBoost**, **Neural Network (NN)** and **Support Vector Machine (SVM)**. We reset the random number seed before each run to ensure that each algorithm will be evaluated using the same data partitions. This means that the results will be directly comparable.

More details can be found within the project.


## <a name="Conclusion"></a> Conclusion
From the experimental section, we see that the most accurate model is the Extra-Tree when using the predictor variables Age, Sex_male, Pclass, Fare, SibSp, Parch, Embarked_Q, Embarked_S, Title_Miss, Title_Mr, Title_Mrs, FamilySizeNote_Singleton, FamilySizeNote_Small_Family and TicketCount. The accuracy of this model on the training set is 0.8388 (+/- 0.02) and on the test set 0.80861.
