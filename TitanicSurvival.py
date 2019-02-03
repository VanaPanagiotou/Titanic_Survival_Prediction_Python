# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:48:17 2018

@author: Vana
"""

# Import libraries

# linear algebra
import numpy as np 

# data processing
import pandas as pd 
pd.set_option('display.max_columns',12)

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# Load training and test data
import os
path="Documents\online teaching\Projects from Internet\Prediction of the survival of passengers in Titanic\Python"
os.chdir(path)



# Making a list of missing value types
missing_values = ["n/a", "na", "--", ""]

train_set = pd.read_csv("train.csv", na_values = missing_values)
test_set = pd.read_csv("test.csv", na_values = missing_values)

# Preview the data
train_set.head()

# Find how many instances (rows) and how many attributes (columns) the data contains
# shape
print(train_set.shape)
# (891, 12)
print(test_set.shape)
# (418, 11)

# The training set has 891 observations and 12 variables and the test set has 
# 418 observations and 11 variables, which means that the traning set has 1 extra variable. 
# Check which variable is missing from the test set. 

colnames_check = np.setdiff1d(train_set.columns.values,test_set.columns.values)
# array(['Survived'], dtype=object)

# As we can see we are missing the "Survived" variable in the test set, 
# which is something that was expected, since we must predict this by creating a model

# more info on the data
print(train_set.info())

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 891 entries, 0 to 890
#Data columns (total 12 columns):
#PassengerId    891 non-null int64
#Survived       891 non-null int64
#Pclass         891 non-null int64
#Name           891 non-null object
#Sex            891 non-null object
#Age            714 non-null float64
#SibSp          891 non-null int64
#Parch          891 non-null int64
#Ticket         891 non-null object
#Fare           891 non-null float64
#Cabin          204 non-null object
#Embarked       889 non-null object
#dtypes: float64(2), int64(5), object(5)
#memory usage: 83.6+ KB
#None


# Statistical Summary
# We can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values and some percentiles.

print(train_set.describe())

#       PassengerId    Survived      Pclass         Age       SibSp  \
#count   891.000000  891.000000  891.000000  714.000000  891.000000   
#mean    446.000000    0.383838    2.308642   29.699118    0.523008   
#std     257.353842    0.486592    0.836071   14.526497    1.102743   
#min       1.000000    0.000000    1.000000    0.420000    0.000000   
#25%     223.500000    0.000000    2.000000   20.125000    0.000000   
#50%     446.000000    0.000000    3.000000   28.000000    0.000000   
#75%     668.500000    1.000000    3.000000   38.000000    1.000000   
#max     891.000000    1.000000    3.000000   80.000000    8.000000   
#
#            Parch        Fare  
#count  891.000000  891.000000  
#mean     0.381594   32.204208  
#std      0.806057   49.693429  
#min      0.000000    0.000000  
#25%      0.000000    7.910400  
#50%      0.000000   14.454200  
#75%      0.000000   31.000000  
#max      6.000000  512.329200  


# We observe that:
# Since the Survived column has dicrete data, the mean gives us the number of 
# people survived from 891 i.e. 38%.
# Most people belonged to Pclass = 3
# The fare prices varied a lot as we can see from the standard deviation of 49

# For categorical variables
print(train_set.describe(include='O'))
#                                                     Name   Sex Ticket Cabin  \
#count                                                 891   891    891   204   
#unique                                                891     2    681   147   
#top     Taylor, Mrs. Elmer Zebley (Juliet Cummins Wright)  male   1601    G6   
#freq                                                    1   577      7     4   
#
#       Embarked  
#count       889  
#unique        3  
#top           S  
#freq        644  


# We observe that:
# The passneger column has two sexes with male being the most common.
# Cabin column has many duplicate values.
# Embarked has three possible values with most passengers embarking from Southhampton.
# Names of all passengers are unique.
# Ticket column also has many duplicate values.


##      Exploratory Analysis

# Class Distribution: Examine what number and what percentage of passengers survived
print(train_set.groupby('Survived').size())
#Survived                       (0=Perished,1=Survived)
#0    549     
#1    342
#dtype: int64


# Class Distribution: Percentage
print(train_set.groupby('Survived').size().apply(lambda x: float(x) / train_set.groupby('Survived').size().sum()*100))
#Survived                       (0=Perished,1=Survived)
#0    61.616162     
#1    38.383838
#dtype: float64

# Class Distribution: Frequency and Percentage
print(pd.DataFrame(data = {'freq': train_set.groupby('Survived').size(), 'percentage':train_set.groupby('Survived').size().apply(lambda x: float(x) / train_set.groupby('Survived').size().sum()*100)}))
#          freq  percentage
#Survived                       (0=Perished,1=Survived)
#0          549   61.616162
#1          342   38.383838


# Find the distribution of people across classes
print(train_set.groupby('Pclass').size())
#Pclass
#1    216
#2    184
#3    491
#dtype: int64


# Check if the passenger class has an impact on survival
print(train_set.groupby('Pclass').mean().Survived)
#Pclass
#1    0.629630
#2    0.472826
#3    0.242363
#Name: Survived, dtype: float64

print(train_set.groupby(['Pclass'])['Survived'].value_counts(normalize=True))
#Pclass  Survived           (0=Perished,1=Survived)
#1       1           0.629630
#        0           0.370370
#2       0           0.527174
#        1           0.472826
#3       0           0.757637
#        1           0.242363
#Name: Survived, dtype: float64


train_set.groupby(['Pclass', 'Survived']).size().reset_index().pivot(columns='Pclass', index='Survived', values=0)
#Pclass      1   2    3
#Survived              
#0          80  97  372
#1         136  87  119


# In Titanic the captain gave order for women and children to be saved first. 
# So, we look into the training set's "Sex" and "Age" variables for any patterns
print(train_set.groupby('Sex').mean().Survived)
#Sex
#female    0.742038
#male      0.188908
#Name: Survived, dtype: float64


print(train_set.groupby(['Sex'])['Survived'].value_counts(normalize=True))
#Sex     Survived
#female  1           0.742038
#        0           0.257962
#male    0           0.811092
#        1           0.188908
#Name: Survived, dtype: float64

# Examine passengers' age
train_set['Age'].describe()
#count    714.000000
#mean      29.699118
#std       14.526497
#min        0.420000
#25%       20.125000
#50%       28.000000
#75%       38.000000
#max       80.000000
#Name: Age, dtype: float64
# There are 177 NA's in Age. But since Age is a very important variable for prediction
# we have to continue the analysis in order to get a better insight from other variables 
# that will help us to tackle the missing values from Age

#### Visual univariate analysis

# Separate the variables into two lists: "cat" for the categorical variables 
# and "cont" for the continuous variables. 
 
cat =  [ 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
cont = ['Fare', 'Age']

# Distribution
fig = plt.figure(figsize=(15, 15))
for i in range (0,len(cat)):
    fig.add_subplot(3,3,i+1)
    sns.countplot(x=cat[i], data=train_set);  

for col in cont:
    fig.add_subplot(3,3,i + 2)
    sns.distplot(train_set[col].dropna());
    i += 1
    
plt.show()
fig.clear()


#### Visual bivariate analysis

# The next charts show the survival (and non-survival) numbers for each variable.

# Count of survival

fig = plt.figure(figsize=(15, 15))
i = 1
for col in cat:
    if col != 'Survived':
        fig.add_subplot(3,3,i)
        sns.countplot(x=col, data=train_set,hue='Survived')
        i += 1

# Box plot Survived vs Age
fig.add_subplot(3,3,6)
sns.swarmplot(x="Survived", y="Age", hue="Sex", data=train_set)
fig.add_subplot(3,3,7)
sns.boxplot(x="Survived", y="Age", data=train_set)

# Survived vs Fare
fig.add_subplot(3,3,8)
sns.violinplot(x="Survived", y="Fare", data=train_set)

# Correlations
corr = train_set.drop(['PassengerId'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig.add_subplot(3,3,9)
sns.heatmap(corr, mask=mask, cmap=cmap, cbar_kws={"shrink": .5})
plt.show()
fig.clear()


# Percentages of survival

fig = plt.figure(figsize=(15, 15))
i = 1
for col in cat:
    if col != 'Survived':
        fig.add_subplot(3,3,i)
        sns.barplot(x=col, data=train_set,y='Survived')
        i += 1

# Survived vs Age
fig.add_subplot(3,3,6)
sns.regplot(x='Age', y='Survived', data=train_set)

# Survived vs Fare
fig.add_subplot(3,3,7)
sns.regplot(x='Fare', y='Survived', data=train_set)

plt.show()
fig.clear()

# Age-Sex-Survived plot
sns.lmplot(x='Age', y='Survived',hue='Sex', data=train_set, palette='Set1')

# From all these plots we observe that:

# There were clearly more male than female on board. 
# The number of female who survived was much more than the males who survived.
#train_set.groupby('Sex',as_index=False).Survived.mean()
#      Sex  Survived
#0  female  0.742038
#1    male  0.188908

# The vast majority of passengers were from the 3rd class.
# Passengers belonging to 1st class had a hude advantage for survival.
#train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#   Pclass  Survived
#0       1  0.629630
#1       2  0.472826
#2       3  0.242363

# Most people embarked from Southampton.
# It seems that the passengers who embarked from Cherbourg had a higher rate of Survival. 
#train_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
#  Embarked  Survived
#0        C  0.553571
#1        Q  0.389610
#2        S  0.336957

# Most passengers didn't have parents/children on board.
# It looks like that passengers who had either 1, 2 or 3 number of parents/children had a higher  
# possibility of surviving than those who had none. However having more than 3 made the 
# possibility even smaller.
#train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#   Parch  Survived
#3      3  0.600000
#1      1  0.550847
#2      2  0.500000
#0      0  0.343658
#5      5  0.200000
#4      4  0.000000
#6      6  0.000000

# Most passengers didn't have spouse/siblings on board.
# It seems that having a spouse or 1 sibling had a positive effect on survival as compared to being alone. 
# However, the chances of survival go down as the number of spouse/siblings increases more than 1.

#train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#   SibSp  Survived
#1      1  0.535885
#2      2  0.464286
#0      0  0.345395
#3      3  0.250000
#4      4  0.166667
#5      5  0.000000
#6      8  0.000000

# Most passengers belong to 20-40 age band.
# Younger individuals were more likely to survive.
# From the boxplot Age-Survived, we can see that there are only a few outliers in Age.
# From the Age-Sex-Survived plot, we see that age has an opposite effect on the survival in men and
# women. The chances of survival increase as the age of women increases. The opposite happens for men,
# i.e. the chances of survival decrease as the age of men increases.

# Fare has a large scale and most of the values are between 0 and 100.
# Passengers who paid more money for their ticket had a higher change of survival.

# As concerning the correlation between the features, we can see that the stronger 
# correlation with Survived is for variables Fare and Pclass. 
# This shows that people from the upper class spent more money (to have a better place),
# and, thus, they had higher probability to survive.



# Check for missing values (empty or NA) in the training set

total = train_set.isnull().sum().sort_values(ascending=False)
percent_1 = train_set.isnull().sum()/train_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
train_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#             Total     %
#Cabin          687  77.1
#Age            177  19.9
#Embarked         2   0.2
#Fare             0   0.0
#Ticket           0   0.0
#Parch            0   0.0
#SibSp            0   0.0
#Sex              0   0.0
#Name             0   0.0
#Pclass           0   0.0
#Survived         0   0.0
#PassengerId      0   0.0

# Check for missing values (empty or NA) in the test set
total = test_set.isnull().sum().sort_values(ascending=False)
percent_1 = test_set.isnull().sum()/test_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
test_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#             Total     %
#Cabin          327  78.2
#Age             86  20.6
#Fare             1   0.2
#Embarked         0   0.0
#Ticket           0   0.0
#Parch            0   0.0
#SibSp            0   0.0
#Sex              0   0.0
#Name             0   0.0
#Pclass           0   0.0
#PassengerId      0   0.0


# We see that we have missing values in Age, Cabin and Embarked in the training set and 
# Age, Fare and Cabin in the test set.
# To tackle this problem, we are going to predict the missing values with the full data set, 
# which means that we need to combine the training and test sets together.

test_set2 = test_set
test_set2['Survived'] = np.nan
# Combine training and test sets 
full_set = train_set.append(test_set2, sort= 'False', ignore_index=True)

# Check for missing values (empty or NA) in the full set (training + test)
total = full_set.isnull().sum().sort_values(ascending=False)
percent_1 = full_set.isnull().sum()/full_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
full_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#             Total     %
#Cabin         1014  77.5
#Survived       418  31.9
#Age            263  20.1
#Embarked         2   0.2
#Fare             1   0.1
#Ticket           0   0.0
#SibSp            0   0.0
#Sex              0   0.0
#Pclass           0   0.0
#PassengerId      0   0.0
#Parch            0   0.0
#Name             0   0.0





# Explore existing variables and create new variables that will help in the prediction




####  Variable "Cabin"
# "Cabin" is missing a lot of its values

# The number of not-null Cabin values
full_set['Cabin'].count()
# 295

# The first character is the deck. 
# In the column 'Cabin', extract the word in the strings and create a Deck variable (A - F)
# by separating and pulling off the deck letter contained in the Cabin
full_set['Deck'] = full_set['Cabin'].str.extract('([a-zA-Z ])', expand=True)


# Find the distribution 
print(full_set.groupby('Deck').size())
#Deck
#A    22
#B    65
#C    94
#D    46
#E    41
#F    21
#G     5
#T     1
#dtype: int64

# We see that there is a Deck value "T", which is invalid. We replace it with NA.

wrong_deck_row = np.where(full_set['Deck'] == 'T')
full_set.loc[full_set.index[wrong_deck_row], 'Deck']  = np.nan

# Then replace all NA values with U (for Unknown)
full_set['Deck'] = full_set['Deck'].fillna('U')
# Since this column has so many missing values, we will not further use it
print(full_set.groupby('Deck').size())
#Deck
#A      22
#B      65
#C      94
#D      46
#E      41
#F      21
#G       5
#U    1015
#dtype: int64




####  Variable "Embarked"

# Find which passengers have missing Embarked variables

embarked_missing_rows = np.where(full_set['Embarked'].isnull() | (full_set['Embarked']==""))
full_set.loc[full_set.index[embarked_missing_rows], :]
#      Age Cabin Embarked  Fare                                       Name  \
#61   38.0   B28      NaN  80.0                        Icard, Miss. Amelie   
#829  62.0   B28      NaN  80.0  Stone, Mrs. George Nelson (Martha Evelyn)   
#
#     Parch  PassengerId  Pclass     Sex  SibSp  Survived  Ticket Deck  
#61       0           62       1  female      0       1.0  113572    B  
#829      0          830       1  female      0       1.0  113572    B  

# We will infer their values for embarkment based on present data that seem relevant: 
# passenger class (Pclass) and fare (Fare)
# We see that passengers who have missing Embarked variables paid $80 and their class is 1

# Get rid of these two passengers with the missing Embarked variables
full_set_without_embarked = full_set.drop(full_set.index[embarked_missing_rows])

# Select only the values of 'Fare' and 'Embarked' for passengers from 1st Class
new_data1 = full_set_without_embarked.loc[full_set_without_embarked['Pclass']==1, ['Fare','Embarked']]

embark_fare = new_data1.groupby('Embarked')[['Fare']].median()
#             Fare
#Embarked         
#C         76.7292
#Q         90.0000
#S         52.0000

# The median fare for a 1st Class passenger departing from Cherbourg ('C') is 76.7292
# The median fare for a 1st Class passenger departing from Queenstown ('Q') is 90.0000
# The median fare for a 1st Class passenger departing from Southampton ('S') is 52.0000

# Since, passengers who have missing Embarked variables paid $80, we can replace their NA
# values with 'C'
full_set.loc[full_set.index[embarked_missing_rows], 'Embarked']  = 'C'




####  Variable "Fare"

full_set['Fare'].describe()

#count    1308.000000
#mean       33.295479
#std        51.758668
#min         0.000000
#25%         7.895800
#50%        14.454200
#75%        31.275000
#max       512.329200
#Name: Fare, dtype: float64

# We see that there is zero fare
full_set.loc[full_set['Fare']==0, ['Age','Fare']]

#       Age  Fare
#179   36.0   0.0
#263   40.0   0.0
#271   25.0   0.0
#277    NaN   0.0
#302   19.0   0.0
#413    NaN   0.0
#466    NaN   0.0
#481    NaN   0.0
#597   49.0   0.0
#633    NaN   0.0
#674    NaN   0.0
#732    NaN   0.0
#806   39.0   0.0
#815    NaN   0.0
#822   38.0   0.0
#1157   NaN   0.0
#1263  49.0   0.0


# There might be some error, since we see that 0 fares are not corresponding to infants,
# that possibly were allowed to travel free of cost.
# Replace 0 fares with the median values of the fares corresponding to each passenger class
# (Pclass) and embarkment (Embarked).

# class_embark_fare
cl_em_fa = full_set.groupby(['Pclass','Embarked'])[['Fare']].median()

#                    Fare
#Pclass Embarked         
#1      C         78.2667 cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 1) & (cl_em_fa.index.get_level_values('Embarked') == 'C')]['Fare']
#       Q         90.0000
#       S         52.0000
#2      C         15.3146
#       Q         12.3500
#       S         15.3750
#3      C          7.8958
#       Q          7.7500
#       S          8.0500


# Replace 0 fare for those departured from Cherbourg ('C')
full_set['Fare'] = np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==1) & (full_set['Embarked'] =='C')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 1) & (cl_em_fa.index.get_level_values('Embarked') == 'C')]['Fare'], 
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==2) & (full_set['Embarked'] =='C')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 2) & (cl_em_fa.index.get_level_values('Embarked') == 'C')]['Fare'],
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==3) & (full_set['Embarked'] =='C')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 3) & (cl_em_fa.index.get_level_values('Embarked') == 'C')]['Fare'],            
         full_set['Fare'])))


# Replace 0 fare for those departured from Queenstown ('Q')     
full_set['Fare'] = np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==1) & (full_set['Embarked'] =='Q')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 1) & (cl_em_fa.index.get_level_values('Embarked') == 'Q')]['Fare'], 
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==2) & (full_set['Embarked'] =='Q')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 2) & (cl_em_fa.index.get_level_values('Embarked') == 'Q')]['Fare'],
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==3) & (full_set['Embarked'] =='Q')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 3) & (cl_em_fa.index.get_level_values('Embarked') == 'Q')]['Fare'],            
         full_set['Fare'])))

# Replace 0 fare for those departured from Southampton ('S')   
full_set['Fare'] = np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==1) & (full_set['Embarked'] =='S')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 1) & (cl_em_fa.index.get_level_values('Embarked') == 'S')]['Fare'], 
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==2) & (full_set['Embarked'] =='S')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 2) & (cl_em_fa.index.get_level_values('Embarked') == 'S')]['Fare'],
         np.where(( (round(full_set['Fare'])==0) & (full_set['Pclass']==3) & (full_set['Embarked'] =='S')), cl_em_fa.loc[(cl_em_fa.index.get_level_values('Pclass') == 3) & (cl_em_fa.index.get_level_values('Embarked') == 'S')]['Fare'],            
         full_set['Fare'])))

# Find in which row Fare is missing
fare_missing_row = np.where(full_set['Fare'].isnull())
full_set.loc[full_set.index[fare_missing_row], :]
#       Age Cabin Embarked  Fare                Name  Parch  PassengerId  \
#1043  60.5   NaN        S   NaN  Storey, Mr. Thomas      0         1044   
#
#      Pclass   Sex  SibSp  Survived Ticket Deck  
#1043       3  male      0       NaN   3701    U 

# We see that the passenger with the missing fare is a 3rd Class passenger, who embarked from
# Southampton ('S').

# Get rid of this passenger with the missing Fare variable
full_set_without_fare = full_set.drop(full_set.index[fare_missing_row])

# Select only the passengers from 3rd Class who departed from Southampton ('S')
new_data2 = full_set_without_fare.loc[(full_set_without_fare['Pclass']==3) & (full_set_without_fare['Embarked']=='S'), ['Fare']]
# and see their median fare value
new_data2.median()
#Fare    8.05
#dtype: float64

# Plot the median fare for passengers from 3rd Class who departed from Southampton ('S')
facet = sns.FacetGrid(full_set[(full_set['Pclass'] == 3) & (full_set['Embarked'] == 'S')], aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, 80))

fare_median = full_set[(full_set['Pclass'] == 3) & (full_set['Embarked'] == 'S')]['Fare'].median()
plt.axvline(x=fare_median, color='r', ls='--')

# Replace the NA Fare value with the median value for their class and embarkment which is $8.05
full_set.loc[full_set.index[fare_missing_row], 'Fare']  = new_data2['Fare'].median()




####  Variable "Ticket"

print(full_set.groupby('Ticket').size())

full_set['Ticket'].value_counts().head()
#CA. 2343    11
#1601         8
#CA 2144      8
#347082       7
#PC 17608     7
#Name: Ticket, dtype: int64

# we see that some passengers have the same ticket number

# Create a new variable "TicketCount" which is the number of passengers that have the same 
# ticket number

# To add frequency back to the original dataframe use transform to return an aligned index
full_set['TicketCount'] = full_set.groupby('Ticket')['Ticket'].transform('count')



# Now, we can create some new variables



####  Create the new variable "Title"

# As seen, the name is a combination of first name, last name and title (eg. Mr, Mrs etc)

# Extract title from passenger names
full_set['Title'] = full_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# summary
full_set['Title'].value_counts()
#Mr          757
#Miss        260
#Mrs         197
#Master       61
#Rev           8
#Dr            8
#Col           4
#Ms            2
#Mlle          2
#Major         2
#Sir           1
#Dona          1
#Mme           1
#Capt          1
#Lady          1
#Jonkheer      1
#Countess      1
#Don           1
#Name: Title, dtype: int64


# Plotting the various titles extracted from the names    
sns.countplot(y='Title', data=full_set) 

# we see that there are many different title groups. We will merge them to the most common 
# 4 groups: 
# Mr and Master for male, 
# Miss and Mrs for female

# Combine small title groups 
full_set['Title'] = full_set['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Mr')

full_set['Title'] = full_set['Title'].replace(['Lady', 'Mlle', 'Mme', 'Ms', 'Countess', 'Dr', 'Dona'], 'Mrs')

# summary
full_set['Title'].value_counts()
#Mr        775
#Miss      260
#Mrs       213
#Master     61
#Name: Title, dtype: int64

# Plotting the final distribution of Title variable
sns.countplot(y='Title', data=full_set)

# Show Title counts by sex 

full_set.groupby(["Sex", "Title"]).size().unstack(fill_value=0)

#Title   Master  Miss   Mr  Mrs
#Sex                           
#female       0   260    0  206
#male        61     0  775    7

# Extract surname from passenger name
full_set['Surname'] = full_set['Name'].str.split(',').str.get(0)

# Find how many unique surnames we have
uniq_surname_size = full_set['Name'].str.split(',').str.get(0).unique().size
#875



####  Create the new variable "FamilySize"
# based on number of siblings/spouse(SibSp) and number of children/parents(Parch).

# Create a family size variable including the passenger themselves
full_set["FamilySize"] = full_set.SibSp + full_set.Parch + 1

# Check if FamilySize has an impact on survival

full_set.groupby(["FamilySize", "Survived"]).size().unstack(fill_value=0)
#Survived    0.0  1.0       # (0=Perished,1=Survived)
#FamilySize          
#1           374  163
#2            72   89
#3            43   59
#4             8   21
#5            12    3
#6            19    3
#7             8    4
#8             6    0
#11            7    0

# Probability table of FamilySize in relation to survival 
full_set.groupby(['FamilySize'])['Survived'].value_counts(normalize=True).unstack(fill_value=0)
#Survived         0.0       1.0     # (0=Perished,1=Survived)
#FamilySize                    
#1           0.696462  0.303538
#2           0.447205  0.552795
#3           0.421569  0.578431
#4           0.275862  0.724138
#5           0.800000  0.200000
#6           0.863636  0.136364
#7           0.666667  0.333333
#8           1.000000  0.000000
#11          1.000000  0.000000

# We see that there's a survival penalty to singletons and those with family sizes above 4. 
# So, we will create a discretized family size variable with 3 levels, since there are 
# comparatively fewer large families.


####  Create a discretized family size variable "FamilySizeNote"

full_set['FamilySizeNote'] = full_set['FamilySize']
full_set.loc[full_set.FamilySize == 1, "FamilySizeNote"] = 'Singleton'
full_set.loc[(full_set.FamilySize >= 2) & (full_set.FamilySize <= 5), "FamilySizeNote"] = "Small_Family"
full_set.loc[full_set.FamilySize >= 5, "FamilySizeNote"] = "Large_Family"



####  Create a "FamilyID" variable 

full_set['FamilyID'] = full_set['Surname'] + '_' + full_set['FamilySize'].astype(str)

# Find how many unique FamilyIDs we have
full_set['FamilyID'].unique().size
#928


# We see that some passengers have the same Surname but are not in the same family
# e.g. two people with surname Andersson that travelled alone and are not in the same family,
# have the same FamilyID, i.e. : Andersson_1

# Create a new variable family.count which is the number of passengers that have the same 
# Surname but are not in the same family

family_count = full_set['Surname'].value_counts()
#head
#Sage         11
#Andersson    11
#Asplund       8
#Goodwin       8
#Davies        7
#Name: Surname, dtype: int64

# In addition, we know that large families might had trouble sticking together in the panic. 
# So let's change the FamilyIDs that have family size of two or less and call it a 
# "small" family. 

FamID = pd.DataFrame(data = {'FamilyID':  full_set['FamilyID'] ,'Freq': full_set.groupby('FamilyID')['FamilyID'].transform('count') })

# Change FamilyIDs for those with small family sizes (<=2)
full_set.loc[FamID['Freq'] <= 2, "FamilyID"] = 'Small'

# Find how many unique FamilyIDs we have now
full_set['FamilyID'].unique().size
#78



####  Variable "Age"

# Replacing the missing values from "Age" with the median age might not be the best idea,
# since the age may differ by groups and categories of the passengers. 

# We are going to predict missing Age variables using two different methods, in order 
# to investigate which one achieves the best results. 


# 1st imputation

# get average, std, and number of NaN values
average_age = full_set['Age'].mean()
std_age = full_set['Age'].std()
count_nan_age = full_set['Age'].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_tmp1 = full_set['Age'].copy()
age_tmp1[np.isnan(age_tmp1)] = rand_age



#  2nd imputation

# We will infer missing values for Age based on present data that seem relevant: 
# passenger class (Pclass), sex (Sex) and title (Title)

# Wwe will group the dataset by Sex, Title and Class, and for each
# subset we will compute the median age.


#age_sex_title_class = full_set.groupby(['Sex','Title', 'Pclass'])[['Age']].median()
#                       Age
#Sex    Title  Pclass      
#female Miss   1       30.0
#              2       20.0
#              3       18.0
#       Mrs    1       45.0
#              2       30.0
#              3       31.0
#male   Master 1        6.0
#              2        2.0
#              3        6.0
#       Mr     1       42.0
#              2       30.0
#              3       26.0
#       Mrs    1       47.0
#              2       38.5
# or with
age_sex_title_class = full_set.pivot_table(values='Age', index=['Title'], columns=['Pclass', 'Sex'], aggfunc=np.median)

#Pclass      1            2            3      
#Sex    female  male female  male female  male
#Title                                        
#Master    NaN   6.0    NaN   2.0    NaN   6.0
#Miss     30.0   NaN   20.0   NaN   18.0   NaN
#Mr        NaN  42.0    NaN  30.0    NaN  26.0
#Mrs      45.0  47.0   30.0  38.5   31.0   NaN

# We see that the median age depends a lot on the Sex, Title and Pclass values

# Define function to return value of this pivot table
def mage(x):
    return age_sex_title_class[x['Pclass']][x['Sex']][x['Title']]

# Replace missing values
age_tmp2=full_set
age_tmp2['Age'].fillna(age_tmp2[age_tmp2['Age'].isnull()].apply(mage, axis=1), inplace=True)



# Compare the original distribution of passenger ages with the predicted 
# in order to select the best prediction and ensure that our prediction
# was correct.


# Plot age distributions
fig1, (axis1,axis2, axis3) = plt.subplots(1,3,figsize=(15,4))
axis1.set_title('Original Age values')
axis2.set_title('New Age values using random values')
axis3.set_title('New Age values using grouped median values')

# plot original Age values
# drop all null values, and convert to int
full_set['Age'].dropna().astype(int).hist(bins=70, ax=axis1, figure=fig1)

# plot imputed Age values
age_tmp1.astype(int).hist(bins=70, ax=axis2, figure=fig1)
age_tmp2['Age'].astype(int).hist(bins=70, ax=axis3, figure=fig1)

# We see that the best prediction is achieved using the second method, so we will use these values
# to replace the missing Age values.

# Replace missing Age values with the predicted 

full_set['Age'] = age_tmp2['Age']

# Show number of missing Age values
full_set['Age'].isnull().sum()
#0



# Now, that we have Age values for all passengers, we can create a few more age-dependent 
# variables




####  Create "Child" variable and indicate whether the passenger is child or adult

# Anyone who is less than 18 years is considered to be child

full_set.loc[full_set['Age'] < 18, 'Child'] = 'Child'
full_set.loc[full_set['Age'] >= 18, 'Child'] = 'Adult'

# Children are mostly likely to be rescued first. Check if this actually happened.
full_set.groupby(["Child", "Survived"]).size().unstack(fill_value=0)

#Survived  0.0  1.0         # (0=Perished,1=Survived)
#Child             
#Adult     495  279
#Child      54   63

# Compute the probability to survive
full_set.groupby(['Child'])['Survived'].value_counts(normalize=True).unstack(fill_value=0)
#Survived       0.0       1.0       # (0=Perished,1=Survived)
#Child                       
#Adult     0.639535  0.360465
#Child     0.461538  0.538462

# There is a ~50% chance that you will survive if you are a child

# Check if female children had a higher chance to survive compared to male children

full_set.groupby(["Child", "Sex", "Survived"]).size().unstack(fill_value=0)

#Survived      0.0  1.0       # (0=Perished,1=Survived)
#Child Sex             
#Adult female   64  195
#      male    431   84
#Child female   17   38
#      male     37   25


# and compute their probability to survive
full_set.groupby(['Child','Sex'])['Survived'].value_counts(normalize=True).unstack(fill_value=0)
#Survived           0.0       1.0        # (0=Perished,1=Survived)
#Child Sex                       
#Adult female  0.247104  0.752896
#      male    0.836893  0.163107
#Child female  0.309091  0.690909
#      male    0.596774  0.403226

# It's obvious that female children are more likely to survive (0.690909) compared to 
# male children (0.403226)



####  Create a "Mother" variable to indicate whether the passenger is Mother or Not Mother
# Mother is a passenger who is female, over 18 years old, has 1 child or more and has the 
# Title "Mrs".

full_set['Mother'] = 'Not Mother'
full_set.loc[(full_set['Sex']=='female') & (full_set['Parch']>0) & (full_set['Age']>18) & (full_set['Title']== 'Mrs'), 'Mother'] = 'Mother'


# Show counts
full_set.groupby(["Mother", "Survived"]).size().unstack(fill_value=0)

#Survived    0.0  1.0             # (0=Perished,1=Survived)
#Mother              
#Mother       16   39
#Not Mother  533  303


# Compute the probability to survive
full_set.groupby(['Mother'])['Survived'].value_counts(normalize=True).unstack(fill_value=0)

#Survived         0.0       1.0             # (0=Perished,1=Survived)
#Mother                        
#Mother      0.290909  0.709091
#Not Mother  0.637560  0.362440

# It's obvious that Mothers are more likely to survive (0.709091) compared to 
# Not Mothers (0.362440)
 




#### Visualizations

# Passengers survived Vs. Passengers Class

from matplotlib.colors import LinearSegmentedColormap
dict=(1,0)
colors = sns.color_palette("Set1", n_colors=len(dict))
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

plt.figure(figsize=(10,10))
sns.countplot(x='Pclass', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], palette=colors[::-1])
plt.xlabel('Passenger Class')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Class in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Passengers Class in percentage

full_set2 = full_set
full_set2['Perished'] = 1 - full_set2['Survived']
full_set2.loc[0:train_set.shape[0],].groupby('Pclass').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Class in Training Set')
plt.xlabel('Passenger Class')
plt.ylabel('Survived')


# Passengers survived Vs. Sex

plt.figure(figsize=(10,10))
sns.countplot(x='Sex', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order=['female', 'male'], palette=colors[::-1])
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Sex in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Sex in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('Sex').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Sex in Training Set')
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# Passengers survived Vs. Age

#fig2, axis1 = plt.subplots(figsize=(10,10))
#axis1.set_title('Survived Passengers Vs. Age in Training Set')
#full_set.loc[0:train_set.shape[0],].dropna().pivot(values='Age', columns='Survived').plot.hist(bins=30, alpha=0.8, ax=axis1, figure=fig2, stacked=True, color=colors[::-1])


figure = plt.figure(figsize=(10, 10))
plt.hist([full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 1]['Age'], full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 0]['Age']], 
         stacked=True, color = colors,
         bins = 50, label = ['Survived','Perished'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.title('Survived Passengers Vs. Age in Training Set')
plt.legend();

# Density Plot
plt.figure(figsize=(10,10))
sns.kdeplot(full_set["Age"][full_set.Survived == 1], color="lightcoral", shade=True)
sns.kdeplot(full_set["Age"][full_set.Survived == 0], color="darkturquoise", shade=True)
plt.legend(['Survived', 'Perished'])
plt.title('Density Plot of Age for Survived and Perished Passengers')
plt.show()

# The age distribution for survivors and perished is actually very similar. 
# One noticeable difference is that, of the survivors, a larger proportion were children.

# Passengers survived Vs. Age + Sex 
# Include Sex in the plot, because it's a significant predictor



## female
#plt.hist([full_set.loc[0:train_set.shape[0]-1,][(full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 1) & (full_set.loc[0:train_set.shape[0]-1,]['Sex'] == 'female')]['Age'], full_set.loc[0:train_set.shape[0]-1,][(full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 0) & (full_set.loc[0:train_set.shape[0]-1,]['Sex'] == 'female')]['Age']], 
#         stacked=True, color = colors,
#         bins = 50, label = ['Survived','Perished'])
#plt.xlabel('Age')
#plt.ylabel('Number of passengers')
#plt.title('Survived Passengers Vs. Age in Training Set')
#plt.legend();

## male
#plt.hist([full_set.loc[0:train_set.shape[0]-1,][(full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 1) & (full_set.loc[0:train_set.shape[0]-1,]['Sex'] == 'male')]['Age'], full_set.loc[0:train_set.shape[0]-1,][(full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 0) & (full_set.loc[0:train_set.shape[0]-1,]['Sex'] == 'male')]['Age']], 
#         stacked=True, color = colors,
#         bins = 50, label = ['Survived','Perished'])
#plt.xlabel('Age')
#plt.ylabel('Number of passengers')
#plt.title('Survived Passengers Vs. Age in Training Set')
#plt.legend();


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 8))
fig.suptitle("Survived Passengers Vs. Age and Sex in Training Set")
women = full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Sex']=='female']
men = full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=40, color = colors[0], label = 'Survived', ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, color = colors[1], label = 'Perished', ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=40, color = colors[0], label = 'Survived',ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, color = colors[1], label = 'Perished', ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')



# Passengers survived Vs. Fare

figure = plt.figure(figsize=(10, 10))
plt.hist([full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 1]['Fare'], full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived'] == 0]['Fare']], 
         stacked=True, color = colors,
         bins = 50, label = ['Survived','Perished'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.title('Survived Passengers Vs. Fare in Training Set')
plt.legend();


# Passengers survived Vs. Fare (Stacked)
plt.hist([full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived']==1]['Fare'], full_set.loc[0:train_set.shape[0]-1,][full_set.loc[0:train_set.shape[0]-1,]['Survived']==0]['Fare']], 
 stacked=True, color = colors, bins = 5,label = ['Survived','Perished'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.title('Survived Passengers Vs. Fare in Training Set')
plt.legend()
plt.show()
# Passengers who paid a higher fare had a higher probability of survival.

# Density Plot
plt.figure(figsize=(15,8))
sns.kdeplot(full_set["Fare"][full_set.Survived == 1], color="lightcoral", shade=True)
sns.kdeplot(full_set["Fare"][full_set.Survived == 0], color="darkturquoise", shade=True)
plt.legend(['Survived', 'Perished'])
plt.title('Density Plot of Fare for Survived and Perished Passengers')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()

# Passengers who paid lower fare appear to have been less likely to survive. 

# Passengers survived Vs. Age and Fare

plt.figure(figsize=(15, 10))
ax = plt.subplot()
ax.scatter(full_set[full_set['Survived'] == 1]['Age'], full_set[full_set['Survived'] == 1]['Fare'], 
           c=colors[0], s=full_set[full_set['Survived'] == 1]['Fare'], label = 'Survived')
ax.scatter(full_set[full_set['Survived'] == 0]['Age'], full_set[full_set['Survived'] == 0]['Fare'], 
           c=colors[1], s=full_set[full_set['Survived'] == 0]['Fare'], label = 'Perished')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
# The size of the circles is proportional to the ticket fare.


# Passengers survived Vs. Embarked

plt.figure(figsize=(10,10))
sns.countplot(x='Embarked', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order=['C', 'Q', 'S'], palette=colors[::-1])
plt.xlabel('Passenger Class')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Embarked in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Embarked in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('Embarked').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Embarked in Training Set')
plt.xlabel('Embarked')
plt.ylabel('Survived')


# Passengers survived Vs. TicketCount

plt.figure(figsize=(10,10))
sns.countplot(x='TicketCount', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], palette=colors[::-1])
plt.xlabel('TicketCount')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. TicketCount in Training Set')
plt.legend(labels={'Perished','Survived'})

# Passengers survived Vs. TicketCount in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('TicketCount').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. TicketCount in Training Set')
plt.xlabel('TicketCount')
plt.ylabel('Survived')
plt.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85),
           bbox_transform=plt.gcf().transFigure)



# Passengers survived Vs. Title

plt.figure(figsize=(10,10))
sns.countplot(x='Title', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order = ['Master', 'Miss', 'Mr', 'Mrs'], palette=colors[::-1])
plt.xlabel('Title')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Title in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Title in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('Title').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Title in Training Set')
plt.xlabel('Title')
plt.ylabel('Survived')



# Passengers survived Vs. FamilySize

plt.figure(figsize=(10,10))
sns.countplot(x='FamilySize', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], palette=colors[::-1])
plt.xlabel('FamilySize')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. FamilySize in Training Set')
plt.legend(labels={'Perished','Survived'})

# Passengers survived Vs. FamilySize in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('FamilySize').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. FamilySize in Training Set')
plt.xlabel('FamilySize')
plt.ylabel('Survived')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)


# Passengers survived Vs. FamilySizeNote

plt.figure(figsize=(10,10))
sns.countplot(x='FamilySizeNote', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order = ['Large_Family', 'Singleton', 'Small_Family'], palette=colors[::-1])
plt.xlabel('Family Size Note')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Family Size Note in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. FamilySizeNote in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('FamilySizeNote').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Family Size Note in Training Set')
plt.xlabel('Family Size Note')
plt.ylabel('Survived')


# Passengers survived Vs. Child/Adult

plt.figure(figsize=(10,10))
sns.countplot(x='Child', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order = ['Adult', 'Child'], palette=colors[::-1])
plt.xlabel('Passenger')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Child/Adult in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Child/Adult in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('Child').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Child/Adult in Training Set')
plt.xlabel('Passenger')
plt.ylabel('Survived')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)


# Passengers survived Vs. Child/Adult + Sex 
# Include Sex in the plot, because it's a significant predictor

g=sns.catplot(x='Child', hue = 'Survived', col= 'Sex', data= full_set.loc[0:train_set.shape[0]-1,], kind = "count", palette = colors[::-1])
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Survived Passengers Vs. Child/Adult and Sex in Training Set')
(g.set_axis_labels("", "Survived"))

g= sns.catplot(x='Child', y='Survived', col='Sex', data= full_set.loc[0:train_set.shape[0]-1,], kind= "bar", ci= None, palette = colors[::-1], legend=True)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Percentage of Survived Passengers Vs. Child/Adult and Sex in Training Set')
(g.set_axis_labels("", "Survived"))



# Passengers survived Vs. Mother

plt.figure(figsize=(10,10))
sns.countplot(x='Mother', hue="Survived", data=full_set.loc[0:train_set.shape[0]-1,], order = ['Mother', 'Not Mother'], palette=colors[::-1])
plt.xlabel('Woman')
plt.ylabel('Survived')
plt.title('Survived Passengers Vs. Mother/Not Mother in Training Set')
plt.legend(labels={'Perished','Survived'})


# Passengers survived Vs. Mother in percentage

full_set2.loc[0:train_set.shape[0]-1,].groupby('Mother').agg('mean')[['Survived', 'Perished']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1)
plt.title('Percentage of Survived Passengers Vs. Mother/Not Mother in Training Set')
plt.xlabel('Woman')
plt.ylabel('Survived')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)







#### PREDICTION

# Encode Categorical features


# Name, Sex, Ticket, Cabin, Embarked, Deck, Title, Surname, FamilySizeNote, FamilyID, Child and Mother
# are categorical variables, but in non-numeric format. We will convert those variables into numeric ones
# so that the classifier can handle them. To do so, we have to find unique classes and encode each class
# a unique integer.

full_set['Name'].unique()
#array(['Braund, Mr. Owen Harris',
#       'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
#       'Heikkinen, Miss. Laina', ..., 'Saether, Mr. Simon Sivertsen',
#       'Ware, Mr. Frederick', 'Peter, Master. Michael J'], dtype=object)

full_set['Sex'].unique()
#array(['male', 'female'], dtype=object)

full_set['Ticket'].unique()
#array(['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450',
#       '330877', '17463', '349909', '347742', '237736', 'PP 9549', '... , 
#       'SC/PARIS 2166', '28666', '334915', '365237', '347086',
#       'A.5. 3236', 'SOTON/O.Q. 3101262', '359309'], dtype=object)

full_set['Cabin'].unique()
#array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
#       'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
#       'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
#       'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
#       'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
#       'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
#       'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
#       'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
#       'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44',
#       'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14',
#       'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38',
#       'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68',
#       'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48',
#       'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
#       'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
#       'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
#       'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
#       'C148', 'B45', 'B36', 'A21', 'D34', 'A9', 'C31', 'B61', 'C53',
#       'D43', 'C130', 'C132', 'C55 C57', 'C116', 'F', 'A29', 'C6', 'C28',
#       'C51', 'C97', 'D22', 'B10', 'E45', 'E52', 'A11', 'B11', 'C80',
#       'C89', 'F E46', 'B26', 'F E57', 'A18', 'E60', 'E39 E41',
#       'B52 B54 B56', 'C39', 'B24', 'D40', 'D38', 'C105'], dtype=object)


full_set['Embarked'].unique()
#array(['S', 'C', 'Q'], dtype=object)

full_set['Deck'].unique()
#array(['U', 'C', 'E', 'G', 'D', 'A', 'B', 'F'], dtype=object)

full_set['Title'].unique()
#array(['Mr', 'Mrs', 'Miss', 'Master'], dtype=object)

full_set['Surname'].unique()
#array(['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen', 'Moran',
#       'McCarthy', 'Palsson', 'Johnson', 'Nasser', 'Sandstrom', 'Bonnell', ...,
#       'Conlon', 'Nourney', 'Riordan', 'Naughton', 'Henriksson',
#       'Spector', 'Oliva y Ocana', 'Saether'], dtype=object)

full_set['FamilySizeNote'].unique()
#array(['Small_Family', 'Singleton', 'Large_Family'], dtype=object)

full_set['FamilyID'].unique()
#array(['Small', 'Palsson_5', 'Johnson_3', 'Sandstrom_3', 'Andersson_7',
#       'Rice_6', 'Williams_1', 'Asplund_7', 'Fortune_6', 'Laroche_4',
#       'Samaan_3', 'Panula_6', 'Harper_2', 'West_4', 'Goodwin_8',
#       'Skoog_6', 'Moubarek_3', 'Caldwell_3', 'Ford_5', 'Dean_4',
#       'Johansson_1', 'Hickman_3', 'Peter_3', 'Boulos_3', 'Navratil_3',
#       'van Billiard_3', 'Sage_11', 'Goldsmith_3', 'Smith_1', 'Klasen_3',
#       'Lefebre_5', 'Becker_4', 'Bourke_3', 'Collyer_3', 'Rosblom_3',
#       'Touma_3', 'Taussig_3', 'Abbott_3', 'Olsson_1', 'Allison_4',
#       'Kelly_1', 'McCoy_3', 'Johnson_1', 'Keane_1', 'Ryerson_5',
#       'Hart_3', 'Nilsson_1', 'Wick_3', 'Spedden_3', 'Coutts_3',
#       'Elias_3', 'Widener_3', 'Nakid_3', 'Carter_4', 'Oreskovic_1',
#       'Drew_3', 'Van Impe_3', 'Danbom_3', 'Flynn_1', 'Dodge_3',
#       'Baclini_4', 'Cacic_1', 'Karlsson_1', 'Svensson_1', 'Quick_3',
#       'Daly_1', 'Crosby_3', 'Davies_3', 'Thayer_3', 'Herman_4', 'Cor_1',
#       'Brown_3', 'Wells_3', 'Carlsson_1', 'Johnston_4', 'Mallet_3',
#       'Compton_3', 'Peacock_3'], dtype=object)

full_set['Child'].unique()
#array(['Adult', 'Child'], dtype=object)

full_set['Mother'].unique()
#array(['Not Mother', 'Mother'], dtype=object)


# FamilyID variable, which may be an important predictor has 78 levels 
# (full_set['FamilyID'].unique().size) 
# We will reduce the number of levels, by compressing all of the families under 3 members into one code

family_ids = full_set.FamilyID.copy()
# We can increase our cut-off to be a "Small" family from 2 to 3 people. 
# Change FamilyIDs for those with small family sizes (<=3)
family_ids[full_set["FamilySize"] <= 3] = 'Small'

# Print the count of each unique id.
pd.value_counts(family_ids)
#Small          1194
#Sage_11          11
#Andersson_7       9
#Goodwin_8         8
#Asplund_7         7
#Skoog_6           6
#Fortune_6         6
#Panula_6          6
#Rice_6            6
#Ryerson_5         5
#Palsson_5         5
#Lefebre_5         5
#Ford_5            5
#Becker_4          4
#Allison_4         4
#Johnston_4        4
#Dean_4            4
#Laroche_4         4
#Baclini_4         4
#West_4            4
#Herman_4          4
#Carter_4          4
#Name: FamilyID, dtype: int64

# Print the new levels of FamilyID
family_ids.unique().size
# 22

full_set["FamilyID2"] = family_ids


# We are going to use get_dummies function which transforms the categorical variables in binary for 
# variables that have only a few categories. 
# The get_dummies function is not recommended if the categorical variables have too many categories,
# as are Cabin, Name, Surname, Ticket and FamilyID variables here.

full_set = pd.get_dummies(full_set,drop_first=True,
                          columns=['Sex','Title','Embarked', 'Deck','FamilySizeNote','Child','Mother', 'FamilyID2'] )





# Scaling numerical variables Age and Fare

from sklearn.preprocessing import StandardScaler 
scale = StandardScaler().fit(full_set[['Age', 'Fare']])
full_set[['Age', 'Fare']] = scale.transform(full_set[['Age', 'Fare']])


# Split the data back into the original training and testing sets

train_set_new = full_set.loc[0:train_set.shape[0]-1,]
test_set_new = full_set.loc[train_set.shape[0]:full_set.shape[0],]


# Set the seed to ensure reproduceability
np.random.seed(1)

#####      Decision Trees


from sklearn.tree import DecisionTreeClassifier

# Build a Decision Tree to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the tree, we first initialize an instance of an untrained decision tree classifier
decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=10, min_samples_leaf=20, random_state=1)
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
decision_tree.fit(x_train, y_train)

# Plot the decision tree to see what were the splits that the model identified 
# as being most significant for the classification

import graphviz
from sklearn.tree import export_graphviz
tree_view = export_graphviz(decision_tree, out_file=None, feature_names = x_train.columns.values, rotate=True) 
tree_viz = graphviz.Source(tree_view)
tree_viz


# Performance on the training set

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Compute the accuracy for all the cross validation folds.  
kfold = KFold(n_splits=10, random_state=1000)
scores = cross_val_score(decision_tree, x_train, y_train, cv=kfold, scoring='accuracy')

# Take the mean of the scores (because we have one for each fold)
print("Model 1 - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
#Model 1 - Accuracy on Training Set: 0.7890 (+/- 0.04)


#The accuracy of the model varies depending on which rows were selected 
#for the training and test sets. For that reason, we will use a shuffle validator.  
#The shuffle validator applies a random split 20:80 for test/training, but it
#also generates 20 (n_splits) unique permutations of this split. In that way, 
#we can test our classifier against each of the different splits.


from sklearn.model_selection import ShuffleSplit

shuffle_validator = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
def test_classifier(model):
    scores = cross_val_score(model, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    print("Model 1 - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

test_classifier(decision_tree)
#Model 1 - Accuracy Training Set: 0.7941 (+/- 0.03)


# Performance on the testing set

y_pred = decision_tree.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('dec_tree1.csv', index=False)
# Kaggle score 0.76555


# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family', # 20
                'TicketCount','FamilyID_factor']
               ]


for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=10, min_samples_leaf=20, random_state=1)
    decision_tree.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(decision_tree, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(decision_tree, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = decision_tree.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="dec_tree%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)



# Summary for Decision Trees - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7890 (+/- 0.04)     0.76555
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.7919 (+/- 0.03)     0.75598
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.7698 (+/- 0.03)     0.76076
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8087 (+/- 0.03)     0.77033
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8075 (+/- 0.03)     0.77033
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8070 (+/- 0.03)     0.77033
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8084 (+/- 0.03)     0.75598
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8106 (+/- 0.02)     0.77033
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.7980 (+/- 0.03)     0.76076
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8028 (+/- 0.03)     0.76076
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8067 (+/- 0.03)     0.76555
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8268 (+/- 0.03)     0.78468
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8268 (+/- 0.03)     0.78468
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8073 (+/- 0.03)     0.77033
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8237 (+/- 0.03)     0.78468                                                             
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8293 (+/- 0.03)     0.78468
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount           0.8293 (+/- 0.03)     0.78468
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F + 
#               Deck_G + Deck_U                                    0.8249 (+/- 0.03)     0.79425 (best)
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount         0.8299 (+/- 0.03)     0.78468
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount +   
#                FamilyID_factor                                   0.8304 (+/- 0.03)     0.78468


# The best prediction for survival with decision trees was achieved using the 18th model.


# We could also tune the decision tree model to find the optimal parameter values.
# We will try using all available predictors.
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    

# Parameters that will be tuned    
# max depth : indicates how deep the tree can be. The deeper the tree, the more splits 
# it has, which means that it captures more information about the data.
max_depth = np.arange(1,10)    
# min_samples_split: represents the minimum number of samples required to split an internal node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = list(range(1,x_train.shape[1])) # 1,46

decision_tree = DecisionTreeClassifier(random_state=1)

param_grid = {'max_depth' : max_depth, 'min_samples_split': min_samples_splits, 
              'min_samples_leaf': min_samples_leaf, 'max_features': max_features}

from sklearn.model_selection import GridSearchCV

import multiprocessing
n_jobs= multiprocessing.cpu_count()-1

decision_tree_cv = GridSearchCV(decision_tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
decision_tree_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(decision_tree_cv.best_params_)
#Best parameters found on training set:
#{'max_depth': 2, 'max_features': 41, 'min_samples_leaf': 0.1, 'min_samples_split': 0.4}
print('Best score:',decision_tree_cv.best_score_)
# Best score: 0.8047138047138047

decision_tree_optimised = decision_tree_cv.best_estimator_
decision_tree_optimised.fit(x_train, y_train)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(decision_tree_optimised)
#Model 1 - Accuracy on Training Set: 0.8025 (+/- 0.02)



# Performance on the testing set
y_pred = decision_tree_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('dec_tree_optimised.csv', index=False)
# Kaggle score 0.76555





#####      Random Forest


# Random Forest fits multiple classification trees to random subsets of the input data and
# averages the predictions to return whichever prediction was returned by the most trees.
# This helps to avoid overfitting, a problem that occurs when a model is so tightly fitted
# to arbitrary correlations in the training set that it performs poorly on the testing set.


from sklearn.ensemble import RandomForestClassifier

# Build a Random Forest to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the Random Forest, we first initialize an instance of an untrained Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1)
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
random_forest.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = random_forest.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')


# 2nd way to plot feature importance

##Features importance
#import collections
#features = x_train.columns.tolist()
#fi = random_forest.feature_importances_
#sorted_features = {}
#for feature, imp in zip(features, fi):
#    sorted_features[feature] = round(imp,3)
#
## sort the dictionnary by value
#sorted_features = collections.OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))
#
##for feature, imp in sorted_features.items():
##print(feature+" : ",imp)
#
#dfvi = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
##dfvi.head()
#plt.figure(figsize=(10, 10))
#sns.barplot(x='Features', y='Importance', data=dfvi);
#plt.xticks(rotation=90) 
#plt.show()


# 3rd way to plot feature importance


## get features importances
#importances = random_forest.feature_importances_
#indices = np.argsort(importances)
#   
## show features importances
#plt.figure(1)
#plt.title('Feature Importance')
#plt.barh(range(len(indices)), importances[indices], align='center')
#plt.yticks(range(len(indices)), x_train.columns)
#plt.xlabel('Relative Importance')


# 4th way to plot feature importance

#import plotly as py
#import plotly.graph_objs as go
#feature_importances = random_forest.feature_importances_
##Plot the importance of each feature
#feature_data = [go.Bar(
#            x=predictors,
#            y=feature_importances
#    )]
#
#feature_layout = go.Layout(autosize = False, width = 400, height = 400,
#                 # yaxis = dict(title = 'Importance'),
#                  title = 'Importance of features')
#fig = go.Figure(data = feature_data, layout = feature_layout)
#
#py.offline.plot(fig) #, filename='feature-importance'



# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(random_forest)
#Model 1 - Accuracy on Training Set: 0.7835 (+/- 0.03)


# Performance on the testing set

y_pred = random_forest.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('r_forest1.csv', index=False)
# Kaggle score 0.72248


      


# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1)
    random_forest.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(random_forest, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(random_forest, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = random_forest.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="r_forest%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)





# Summary for Decision Trees - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7835 (+/- 0.03)     0.72248  
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8209 (+/- 0.03)     0.72727
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.8061 (+/- 0.03)     0.75598
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8402 (+/- 0.03)     0.75119
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8419 (+/- 0.03)     0.75119
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8405 (+/- 0.03)     0.76076
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8296 (+/- 0.03)     0.73205  
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8257 (+/- 0.02)     0.77511
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8299 (+/- 0.02)     0.77033 
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8277 (+/- 0.03)     0.74641 
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8257 (+/- 0.03)     0.75119
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8436 (+/- 0.03)     0.75598    
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8455 (+/- 0.03)     0.77033
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8430 (+/- 0.03)     0.78947
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8441 (+/- 0.03)     0.78468                                                         
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8466 (+/- 0.02)     0.78468
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount          0.8430 (+/- 0.03)      0.77990
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F + 
#               Deck_G + Deck_U                                    0.8394 (+/- 0.03)     0.79904  (best)
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount         0.8453 (+/- 0.03)     0.77990
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                             0.8430 (+/- 0.02)     0.79904  (best)



# The best prediction for survival with random forests was achieved using the 18th and the 20th models.


# We could also tune the random forest model to find the optimal parameter values.
# We will try using all available predictors,
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    
 
# Parameters that will be tuned   
 
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]
# criterion: Gini is intended for continuous attributes and Entropy for attributes that occur 
# in classes. Gini tends to find the largest class, while Entropy tends to find groups of classes 
# that make up ~50% of the data. Gini minimizes misclassification. 
# Entropy may be a little slower to compute because it makes use of the logarithm.
criterion = ['gini', 'entropy']


random_forest = RandomForestClassifier(random_state=1)

param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,  
              'min_samples_split': min_samples_splits, 'min_samples_leaf': min_samples_leaf, 
              'max_features': max_features, 'criterion': criterion}


random_forest_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
random_forest_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(random_forest_cv.best_params_)
#Best parameters found on training set:
#{'criterion': 'gini', 'max_depth': 2, 'max_features': 0.5, 'min_samples_leaf': 0.2, 
#'min_samples_split': 0.5, 'n_estimators': 35}
print('Best score:',random_forest_cv.best_score_)
# Best score: 0.8024691358024691

random_forest_optimised = random_forest_cv.best_estimator_
random_forest_optimised.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = random_forest_optimised.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(random_forest_optimised)
#Model 1 - Accuracy on Training Set: 0.8008 (+/- 0.03)



# Performance on the testing set
y_pred = random_forest_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('random_forest_optimised.csv', index=False)
# Kaggle score 0.77511





# Extra-Tree method (Extremely Randomized trees)

#Extremely Randomized trees are very similar to Random Forests. Their main differences are that
#Extremely Randomized trees do not resample observations when building a tree (they do not perform bagging)
#and do not use the "best split". The resulting "forest" contains trees that are more variable, 
#but less correlated than the trees in a Random Forest, which means that Extremely Randomized trees 
#are better than Random Forest in term of variance. 


from sklearn.ensemble import ExtraTreesClassifier


# Build Extremely Randomized trees to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the Extremely Randomized trees, we first initialize an instance of an untrained Extremely 
# Randomized trees classifier
extra_tree = ExtraTreesClassifier(n_estimators=100, max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1)


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
extra_tree.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = extra_tree.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')


# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(extra_tree)
#Model 1 - Accuracy on Training Set: 0.8003 (+/- 0.03)


# Performance on the testing set

y_pred = extra_tree.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('extra_tree1.csv', index=False)
# Kaggle score 0.75598





# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    extra_tree = ExtraTreesClassifier(n_estimators=100, max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1)
    extra_tree.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(extra_tree, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(extra_tree, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = extra_tree.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="extra_tree%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)





# Summary for Extremely Randomized Trees - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.8003 (+/- 0.03)     0.75598
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8142 (+/- 0.03)     0.75119
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.8000 (+/- 0.02)     0.76555
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8226 (+/- 0.03)     0.77033
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8257 (+/- 0.02)     0.77511
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8268 (+/- 0.02)     0.78468
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8285 (+/- 0.02)     0.77990
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8285 (+/- 0.03)     0.78947
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8344 (+/- 0.02)     0.78947
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8324 (+/- 0.03)     0.77990
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8260 (+/- 0.02)     0.78468 
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8257 (+/- 0.03)     0.77511  
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8349 (+/- 0.02)     0.78468 
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8299 (+/- 0.03)     0.77990
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8419 (+/- 0.03)     0.80382                                                     
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount         0.8388 (+/- 0.02)      0.80861 (best)
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount          0.8374 (+/- 0.02)      0.79425
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                    0.8399 (+/- 0.03)      0.78947
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount         0.8385 (+/- 0.02)      0.79904
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                             0.8411 (+/- 0.02)     0.78947



# The best prediction for survival with extremely randomized tress was achieved using the 16th model.




# We could also tune the extremely randomized trees model to find the optimal parameter values.
# We will try using all available predictors,
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    
 
# Parameters that will be tuned   
 
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]
# criterion: Gini is intended for continuous attributes and Entropy for attributes that occur 
# in classes. Gini tends to find the largest class, while Entropy tends to find groups of classes 
# that make up ~50% of the data. Gini minimizes misclassification. 
# Entropy may be a little slower to compute because it makes use of the logarithm.
criterion = ['gini', 'entropy']


extra_tree = ExtraTreesClassifier(random_state=1)

param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,  
              'min_samples_split': min_samples_splits, 'min_samples_leaf': min_samples_leaf, 
              'max_features': max_features, 'criterion': criterion}


extra_tree_cv = GridSearchCV(extra_tree, param_grid=param_grid, cv=5, scoring='accuracy', verbose=150, n_jobs = n_jobs)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
extra_tree_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(extra_tree_cv.best_params_)
#Best parameters found on training set:
#{'criterion': 'gini', 'max_depth': 2, 'max_features': 'log2', 'min_samples_leaf': 0.1, 
# 'min_samples_split': 0.30000000000000004, 'n_estimators': 20}
print('Best score:',extra_tree_cv.best_score_)
# Best score: 0.8159371492704826

extra_tree_optimised = extra_tree_cv.best_estimator_
extra_tree_optimised.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = extra_tree_optimised.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(extra_tree_optimised)
#Model 1 - Accuracy on Training Set: 0.8243 (+/- 0.02)


# Performance on the testing set
y_pred = extra_tree_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('extra_tree_optimised.csv', index=False)
# Kaggle score 0.78468






#####      Logistic Regression
##    Generalized Linear Models (GLMs)


from sklearn.linear_model import LogisticRegression

# Build a Logistic Regression model to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the Logistic Regression, we first initialize an instance of an 
# untrained Logistic Regression classifier
log_reg = LogisticRegression(solver ='lbfgs', random_state=1) # penalty = 'l2', C = 1


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
log_reg.fit(x_train, y_train)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(log_reg)
#Model 1 - Accuracy on Training Set: 0.7966 (+/- 0.02)


# Performance on the testing set

y_pred = log_reg.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('log_reg1.csv', index=False)
# Kaggle score 0.76555



# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    log_reg = LogisticRegression(solver ='lbfgs', random_state=1) # penalty = 'l2', C = 1
    log_reg.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(log_reg, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(log_reg, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = log_reg.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="log_reg%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)

                 


# Summary for Logistic Regression - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7966 (+/- 0.02)    0.76555
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8067 (+/- 0.02)    0.75119
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.7933 (+/- 0.03)    0.76076
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8067 (+/- 0.02)    0.75119
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8179 (+/- 0.02)    0.76555
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8193 (+/- 0.02)    0.76555
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8184 (+/- 0.02)    0.76076
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8008 (+/- 0.02)    0.77033
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8131 (+/- 0.02)    0.77033
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8313 (+/- 0.02)    0.76076
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8184 (+/- 0.02)    0.76555
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8064 (+/- 0.02)    0.77033
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8344 (+/- 0.02)    0.77990
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8151 (+/- 0.02)    0.76555
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8391 (+/- 0.03)    0.78468    
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8372 (+/- 0.03)    0.77990
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount           0.8383 (+/- 0.03)    0.78468
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                     0.8405 (+/- 0.03)    0.78947 (best)
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8402 (+/- 0.02)    0.77990
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                              0.8405 (+/- 0.02)    0.77990



# The best prediction for survival with Logistic Regression was achieved using the 18th model.



# We could also tune the Logistic Regression model to find the optimal parameter values.
# We will try using all available predictors,
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    
 
# Parameters that will be tuned   
# penalty: Regularization is a way to avoid overfitting by penalizing high-valued regression coefficients.
# It can be used to train models that generalize better on unseen data, by preventing the algorithm 
# from overfitting the training dataset. 
# A regression model that uses L1 regularization technique is called Lasso Regression and a model 
# which uses L2 is called Ridge Regression.
# The key difference between these techniques is that Lasso shrinks the less important feature’s 
# coefficient to zero, thus, removing some feature completely. So, this works well for feature selection 
# in case we have a huge number of features.
# We will use Solver 'lbfgs' which supports only l2 penalties. 
# So we can't try to tune the penalty parameter here.
# penalty = ['l1','l2']
# C: C = 1/λ
# Lambda (λ) controls the trade-off between allowing the model to increase its complexity as much as 
# it wants while trying to keep it simple. For example, if λ is very low or 0, the model will have 
# enough power to increase its complexity (overfit) by assigning big values to the weights for 
# each parameter. If, in the other hand, we increase the value of λ, the model will tend to underfit, 
# as the model will become too simple.
# Parameter C will work the other way around. For small values of C, we increase the regularization 
# strength which will create simple models and thus underfit the data. For big values of C, 
# we reduce the power of regularization, which imples the model is allowed to increase its complexity, 
# and therefore, overfit the data.
C = [0.0001, 0.001, 0.01, 0.05, 0.09, 1, 2, 3, 4, 5, 10, 100]  


log_reg = LogisticRegression(solver ='lbfgs', random_state=1)

param_grid = {'C': C}


log_reg_cv = GridSearchCV(log_reg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=150, n_jobs = n_jobs)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
log_reg_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(log_reg_cv.best_params_)
#Best parameters found on training set:
#{'C': 1}
print('Best score:', log_reg_cv.best_score_)
# Best score: 0.8338945005611672

log_reg_optimised = log_reg_cv.best_estimator_
log_reg_optimised.fit(x_train, y_train)


# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(log_reg_optimised)
#Model 1 - Accuracy on Training Set: 0.8388 (+/- 0.03)



# Performance on the testing set
y_pred = log_reg_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('log_reg_optimised.csv', index=False)
# Kaggle score 0.79425







#####      Gradient Boosting Machine (GBM)


# The Gradient Boosting classifier generates many weak prediction trees and combines or "boost" them 
# into a stronger model. 

from sklearn.ensemble import GradientBoostingClassifier


# Build a Gradient Boosting Classifier to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the Gradient Boosting Classifier, we first initialize an instance of an 
# untrained Gradient Boosting Classifier
gbm = GradientBoostingClassifier(max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1) 


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
gbm.fit(x_train, y_train)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(gbm)
#Model 1 - Accuracy on Training Set: 0.7788 (+/- 0.02)


# Performance on the testing set

y_pred = gbm.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('gbm1.csv', index=False)
# Kaggle score 



# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    gbm = GradientBoostingClassifier(max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1) 
    gbm.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(gbm, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(gbm, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = gbm.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="gbm%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)

                 


# Summary for Gradient Boosting Machine - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7788 (+/- 0.02)    0.76076 (best)
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8117 (+/- 0.04)    0.71291
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.7911 (+/- 0.02)    0.71770
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8237 (+/- 0.02)    0.73684
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8235 (+/- 0.02)    0.69377
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8168 (+/- 0.03)    0.71770
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8137 (+/- 0.03)    0.69856
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.7947 (+/- 0.02)    0.71770
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8078 (+/- 0.02)    0.74641 
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8268 (+/- 0.03)    0.71291
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8056 (+/- 0.03)    0.74641
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8232 (+/- 0.03)    0.71770
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8221 (+/- 0.03)    0.70813
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8237 (+/- 0.03)    0.71291
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8240 (+/- 0.03)    0.73205
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8134 (+/- 0.03)    0.75119
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount           0.8123 (+/- 0.02)    0.72248
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                     0.8229 (+/- 0.02)    0.73205
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8156 (+/- 0.03)    0.74641
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                             0.8285 (+/- 0.02)    0.75119



# The best prediction for survival with Gradient Boosting Machine was achieved using the 1st model.


# We could also tune the Gradient Boosting Machine model to find the optimal parameter values.
# We will try using all available predictors,
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    
 
# Parameters that will be tuned  

# learning_rate: learning rate shrinks the contribution of each tree by learning_rate.
# There is a trade-off between learning_rate and n_estimators.
learning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
# Gradient boosting is fairly robust to over-fitting so a large number usually
# results in better performance.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# loss: loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for 
# classification with probabilistic outputs. For loss 'exponential' gradient boosting
# recovers the AdaBoost algorithm.
loss = ['deviance','exponential']
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]

gbm = GradientBoostingClassifier(random_state=1)

param_grid = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'loss': loss, 
              'max_depth': max_depth, 'min_samples_split': min_samples_splits, 
              'min_samples_leaf': min_samples_leaf, 'max_features': max_features}


gbm_cv = GridSearchCV(gbm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=150, n_jobs = n_jobs)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
gbm_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(gbm_cv.best_params_)
#Best parameters found on training set:
#{'learning_rate': 0.25, 'loss': 'deviance', 'max_depth': 10, 'max_features': 0.5, 
# 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 150}
print('Best score:',gbm_cv.best_score_)
# Best score: 0.8484848484848485


gbm_optimised = gbm_cv.best_estimator_
gbm_optimised.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = gbm_optimised.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)


# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(gbm_optimised)
#Model 1 - Accuracy on Training Set: 0.8461 (+/- 0.02)


# Performance on the testing set
y_pred = gbm_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('gbm_optimised.csv', index=False)
# Kaggle score 0.76555








########   XGBoost (or eXtreme Gradient Boosting)  

#XGBoost is one of the implementations of Gradient Boosting concept.Its difference
#with the classical Gradient Boosting is that it uses a more regularized model 
#formalization to control over-fitting, which gives it better performance.


from xgboost import XGBClassifier


# Build a XGBoost Classifier to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the XGBoost Classifier, we first initialize an instance of an 
# untrained XGBoost Classifier

xgb = XGBClassifier(n_estimators=100, max_depth=10, gamma=0, random_state=1) 


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
xgb.fit(x_train, y_train)


# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = xgb.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')


# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(xgb)
#Model 1 - Accuracy on Training Set: 0.7832 (+/- 0.02)


# Performance on the testing set

y_pred = xgb.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('xgb1.csv', index=False)
# Kaggle score 




# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    xgb = XGBClassifier(n_estimators=100, max_depth=10, gamma=0, random_state=1) 

    xgb.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(xgb, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(xgb, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = xgb.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="xgb%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)

                 


# Summary for XGBoost - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7832 (+/- 0.02)    0.74641
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8235 (+/- 0.03)    0.72727
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.8098 (+/- 0.03)    0.76555
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8366 (+/- 0.02)    0.76555
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8335 (+/- 0.03)    0.74641
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8360 (+/- 0.03)    0.75119
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8316 (+/- 0.03)    0.74162
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8109 (+/- 0.03)    0.73684
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8154 (+/- 0.03)    0.74641
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8352 (+/- 0.03)    0.75119
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8265 (+/- 0.03)    0.74162
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8358 (+/- 0.03)    0.77033 (best)
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8310 (+/- 0.03)    0.75598
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8346 (+/- 0.03)    0.74641
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8307 (+/- 0.03)    0.73684
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8299 (+/- 0.03)    0.75598
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount           0.8296 (+/- 0.03)    0.75119
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                     0.8277 (+/- 0.03)    0.72248
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8307 (+/- 0.03)    0.75119
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                              0.8349 (+/- 0.03)    0.75119



# The best prediction for survival with XGBoost was achieved using the 12th model.




# We could also tune the XGBoost model to find the optimal parameter values.
# We will try using all available predictors,
predictor=list(train_set_new)
predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
predictor= list(set(predictor).difference(set(predictors_to_remove)))
x_train = train_set_new.loc[:,predictor]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictor]
    
 
# Parameters that will be tuned  
# learning_rate: learning rate shrinks the contribution of each tree by learning_rate.
# There is a trade-off between learning_rate and n_estimators.
learning_rate = [0.01, 0.02, 0.05, 0.1, 0.2]
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
# XGBoost is fairly robust to over-fitting so a large number usually
# results in better performance.
n_estimators = [100, 200, 300, 500, 1000]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [3,7,10]
# gamma: represents the minimum loss reduction required to make a further
# partition on a leaf node of the tree.
gamma = [0, 0.1, 0.2, 0.5, 1]
# reg_alpha : (xgb's alpha) L1 regularization term on weights
reg_alpha = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
# min_child_weight : Minimum sum of instance weight (hessian) needed in a child.
min_child_weight = [1,3,5]


xgb = XGBClassifier(random_state=1)

param_grid = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 
              'max_depth': max_depth, 'gamma': gamma, 'reg_alpha': reg_alpha, 
              'min_child_weight': min_child_weight}


xgb_cv = GridSearchCV(xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=150, n_jobs = n_jobs)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
xgb_cv.fit(x_train, y_train)


print("Best parameters found on training set:")
print(xgb_cv.best_params_)
#Best parameters found on training set:
# {'gamma': 0, 'learning_rate': 0.02, 'max_depth': 7, 'min_child_weight': 3, 
# 'n_estimators': 500, 'reg_alpha': 0.005}
print('Best score:',xgb_cv.best_score_)
# Best score: 0.8462401795735129


xgb_optimised = xgb_cv.best_estimator_
xgb_optimised.fit(x_train, y_train)

# Plot the relative variable importance

features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = xgb_optimised.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)



# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(xgb_optimised)
#Model 1 - Accuracy on Training Set: 0.8441 (+/- 0.03)



# Performance on the testing set
y_pred = xgb_optimised.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('xgb_optimised.csv', index=False)
# Kaggle score 0.75119




#####      Neural Network (NN)


# Build a Neural Network Classifier to predict Survived using the variables Age and Sex
predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow import set_random_seed



# Create a NN with one input layer with x_train.shape[1] nodes which feeds into a hidden layer with 8 nodes 
# and an output layer which is used to predict a passenger's survival.
# The output layer has a sigmoid activation function, which is used to 'squash' all the outputs 
# to be between 0 and 1.

## Function to create model, required for KerasClassifier
def create_model(hidden_layer_sizes=[8], act='linear', opt='Adam', dr=0.0):
    
    # set random seed for reproducibility
    set_random_seed(42)
    
    # Initialising the NN
    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(hidden_layer_sizes[0], input_dim=x_train.shape[1], activation=act)) 
    
    # create additional hidden layers
    for i in range(1,len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], activation=act))
    
    # add dropout (default is none)
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    
    # Compiling the NN
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

model = create_model()
print(model.summary())

#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_343 (Dense)            (None, 8)                 24        
#_________________________________________________________________
#dropout_167 (Dropout)        (None, 8)                 0         
#_________________________________________________________________
#dense_344 (Dense)            (None, 1)                 9         
#=================================================================
#Total params: 33
#Trainable params: 33
#Non-trainable params: 0
#_________________________________________________________________
#None




# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])
#Training loss: 0.5739783787165427
#Training accuracy: 0.6947250273225016


# Performance on the testing set

y_pred1 = model.predict(x_test)
y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('nn1.csv', index=False)
# Kaggle score 0.69377



# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    model = create_model()
    # Next we “fit” this classifier to our training set, enabling it to learn about
    # how different factors affect the survivability of a passenger
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    # Performance on the training set
        
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Model %d - Training loss: %0.4f'  % (idx+2, score[0]))
    print('Model %d - Training accuracy: %0.4f' % (idx+2, score[1]))

    
    # Performance on the testing set
    y_pred1 = model.predict(x_test)
    y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="nn%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)

                 




# Summary for Neural Network - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                                  0.6947     0.69377
# 
# 2. Survived ~ Age + Sex_male + Pclass                                         0.7935     0.75119
# 
# 3. Survived ~ Age + Sex_male + Fare                                           0.7823     0.76076
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                                  0.7834     0.76076
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch                  0.7890     0.76076
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                                         0.7868     0.77033
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                            0.7901     0.76555
# 
# 8. Survived ~ Age + Sex_male + FamilySize                                     0.7868     0.76555
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                                     0.7868     0.76555
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                                     0.8103     0.77990
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                                         0.7935     0.77033
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                               0.7991     0.77511
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                              0.8002     0.77511
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                                0.7935     0.76555
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                                    0.8272      0.78468 (best)
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount                      0.8092      0.77990
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount                       0.8260      0.78468  (best)
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                                 0.8328     0.77990
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount                      0.8114     0.77511
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                                          0.8137     0.76555



# The best prediction for survival with Neural Network was achieved using the 15th and 17 models.

#
## We could also tune the Neural Network  model to find the optimal parameter values.
## We will use the predictors of the 15th model
#predictor = ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother',
#                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
#                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family']
#x_train = train_set_new.loc[:,predictor]
#y_train = train_set_new.loc[:,'Survived']
#x_test = test_set_new.loc[:,predictor]
#    
#
#
## Parameters that will be tuned 
# 
## define the grid search parameters
##size of data to grab at a time
#batch_size = [16, 32, 64]
##loops on dataset
#epochs = [50, 100]
## optimizer
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
## hidden neurons
## # of neurons
## neurons = [[8],[10],[12],[18]]
#hidden_layer_sizes = [[8],[10],[10,5],[12,6],[12,8,4]]
## dropout
#dr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
#
#param_grid = {'batch_size': batch_size, 'epochs': epochs, 'opt': optimizer, 
#              'hidden_layer_sizes': hidden_layer_sizes, 'dr': dr}
#
#
#from keras.wrappers.scikit_learn import KerasClassifier
#
### Since running GridSearch may be quite time/calculation consuming  
## # we will use GPU based tensorflow in order to speed-up calculation  
##import tensorflow as tf  
##from keras.backend.tensorflow_backend import set_session  
##
##config = tf.ConfigProto(log_device_placement=True)  
### Defining GPU usage limit to 75%  
##config.gpu_options.per_process_gpu_memory_fraction = 0.75  
### Inserting session configuration   
##ses=set_session(tf.Session(config=config))  
#
## create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
#
#
## search the grid
#
#nn_cv = GridSearchCV(estimator=model,  param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
## Set the verbose parameter in GridSearchCV to a positive number 
## (the greater the number the more detail you will get) to print out progress
#nn_cv.fit(x_train, y_train)
#
#
#print("Best parameters found on training set:")
#print(nn_cv.best_params_)
##Best parameters found on training set:
##
#print('Best score:',nn_cv.best_score_)
## Best score: 
#
#
#nn_optimised = nn_cv.best_estimator_
#nn_optimised.fit(x_train, y_train)
#
#
#
## Performance on the training set
#score = nn_optimised.evaluate(x_train, y_train, verbose=0)
#print('Optimised Model - Training loss: %0.4f'  % (score[0]))
#print('Optimised Model - Training accuracy: %0.4f' % (score[1]))
#
#
## Performance on the testing set
#y_pred1 = nn_optimised.predict(x_test)
#y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
## Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
#my_solution  = pd.DataFrame({
#        "PassengerId": test_set_new.loc[:,'PassengerId'],
#        "Survived": y_pred
#    })
#my_solution.to_csv('nn_optimised.csv', index=False)
## Kaggle score 
#
#
#





#####      Support Vector Machines (SVM) with Radial Basis Function (RBF) Kernel 


from sklearn.svm import SVC


# Build a Support Vector Machine to predict Survived 
# using the variables Age and Sex

predictors = ['Age','Sex_male']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]

# To create the SVM Classifier, we first initialize an instance of an 
# untrained SVM Classifier
svm = SVC(C=1, gamma='auto', kernel='rbf', # default values
           random_state=1) 

# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
svm.fit(x_train, y_train)

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(svm)
#Model 1 - Accuracy on Training Set: 0.7986 (+/- 0.03)


# Performance on the testing set

y_pred = svm.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('svm1.csv', index=False)
# Kaggle score 0.75598



# Create a list of predictors

predictors = [ ['Age','Sex_male','Pclass'], # 2
               ['Age','Sex_male','Fare'], # 3
               ['Age','Sex_male','Pclass','Fare'], # 4
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch'], # 5
               ['Age','Sex_male','Pclass','SibSp','Parch','Fare','Embarked_Q','Embarked_S'], # 6
               ['Age','Sex_male','Pclass','FamilySize'], # 7
               ['Age','Sex_male','FamilySize'], # 8
               ['Age','Sex_male','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 9
               ['Age','Sex_male','Pclass','FamilySizeNote_Singleton','FamilySizeNote_Small_Family'], # 10
               ['Age','Sex_male','Pclass','FamilySize','Embarked_Q','Embarked_S'], # 11
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs'], # 12
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Title_Miss','Title_Mr','Title_Mrs'], # 13
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother'], # 14
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 15
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', # 16
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount'],
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 17
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount'], 
               ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Child_Child','Mother_Not Mother', # 18
                'Embarked_Q','Embarked_S','Title_Miss','Title_Mr','Title_Mrs',
                'FamilySizeNote_Singleton','FamilySizeNote_Small_Family','TicketCount',
                'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G', 'Deck_U'],
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs',
                'Embarked_Q','Embarked_S','FamilySizeNote_Singleton','FamilySizeNote_Small_Family',
                'TicketCount'], # 19
               ['Age','Sex_male','Pclass','Fare','Title_Miss','Title_Mr','Title_Mrs', # 20
                'Embarked_Q','Embarked_S','FamilySize', 
                'FamilyID2_Andersson_7', 'FamilyID2_Asplund_7', 'FamilyID2_Baclini_4',
                'FamilyID2_Becker_4', 'FamilyID2_Carter_4', 'FamilyID2_Dean_4',
                'FamilyID2_Ford_5', 'FamilyID2_Fortune_6', 'FamilyID2_Goodwin_8',
                'FamilyID2_Herman_4', 'FamilyID2_Johnston_4', 'FamilyID2_Laroche_4',
                'FamilyID2_Lefebre_5', 'FamilyID2_Palsson_5', 'FamilyID2_Panula_6',
                'FamilyID2_Rice_6', 'FamilyID2_Ryerson_5', 'FamilyID2_Sage_11',
                'FamilyID2_Skoog_6', 'FamilyID2_Small', 'FamilyID2_West_4',
                 'SibSp','Parch']
               ]



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Survived']
    x_test = test_set_new.loc[:,predictor]
    
    svm = SVC(C=1, gamma='auto', kernel='rbf', # default values
                                       random_state=1) 
    svm.fit(x_train, y_train)
    
    # Performance on the training set
    
#    kfold = KFold(n_splits=10, random_state=1000)
#    scores = cross_val_score(svm, x_train, y_train, cv=kfold, scoring='accuracy')
#
#    # Take the mean of the scores (because we have one for each fold)
#    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
#    
    scores = cross_val_score(svm, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+2, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = svm.predict(x_test).astype(int)
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    filename="svm%d.csv" % (idx+2)
    my_solution  = pd.DataFrame({
            "PassengerId": test_set_new.loc[:,'PassengerId'],
            "Survived": y_pred
            })
    my_solution.to_csv(filename, index=False)

                 

# Summary for SVM - Accuracy  


#    Formula                                                  Accuracy (Training Set)    Accuracy (Test Set)
# 1. Survived ~ Age + Sex_male                                      0.7986 (+/- 0.03)    0.75598
# 
# 2. Survived ~ Age + Sex_male + Pclass                             0.8075 (+/- 0.02)    0.76555
# 
# 3. Survived ~ Age + Sex_male + Fare                               0.7975 (+/- 0.02)    0.77033
# 
# 4. Survived ~ Age + Sex_male + Pclass + Fare                      0.8162 (+/- 0.02)    0.76555
# 
# 5. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch      0.8338 (+/- 0.02)    0.78468
# 
# 6. Survived ~ Age + Sex_male + Pclass + SibSp + Parch + Fare +  
#               Embarked_Q + Embarked_S                             0.8335 (+/- 0.02)    0.79904 (best)
# 
# 7. Survived ~ Age + Sex_male + Pclass + FamilySize                0.8360 (+/- 0.02)    0.78468
# 
# 8. Survived ~ Age + Sex_male + FamilySize                         0.8316 (+/- 0.02)    0.78468
# 
# 9. Survived ~ Age + Sex_male + FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family                         0.8327 (+/- 0.02)    0.78947
# 
# 10. Survived ~ Age + Sex_male + Pclass + FamilySizeNote_Singleton
#               FamilySizeNote_Small_Family                         0.8416 (+/- 0.02)    0.78947
# 
# 11. Survived ~ Age + Sex_male + Pclass + FamilySize + 
#               Embarked_Q + Embarked_S                             0.8332 (+/- 0.02)    0.79425
# 
# 12. Survived ~ Age + Sex_male + Pclass + Fare + 
#               Title_Miss + Title_Mr + Title_Mrs                   0.8137 (+/- 0.03)    0.78468
# 
# 13. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Title_Miss + Title_Mr + Title_Mrs                  0.8385 (+/- 0.02)    0.78947
# 
# 14. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother                    0.8352 (+/- 0.03)    0.79425
# 
# 15. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Child_Child + Mother_Not Mother + Embarked_Q + 
#                Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family                        0.8380 (+/- 0.02)    0.78947
# 
# 16.  Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#                Embarked_Q + Embarked_S + Title_Miss + Title_Mr + 
#                Title_Mrs + FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8411 (+/- 0.02)    0.78947
#                                                                   
# 17. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch +
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs + 
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount           0.8411 (+/- 0.02)    0.78947
#                                                                   
# 18. Survived ~ Age + Sex_male + Pclass + Fare + SibSp + Parch + 
#               Child_Child + Mother_Not Mother + Embarked_Q + 
#               Embarked_S + Title_Miss + Title_Mr + Title_Mrs +
#               FamilySizeNote_Singleton + 
#               FamilySizeNote_Small_Family + TicketCount +  
#               Deck_B + Deck_C + Deck_D + Deck_E + Deck_F +  
#               Deck_G + Deck_U                                     0.8405 (+/- 0.02)    0.78947
#                                                              
# 19. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss + 
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S +
#                FamilySizeNote_Singleton + 
#                FamilySizeNote_Small_Family + TicketCount          0.8430 (+/- 0.02)    0.79425
#                                                                 
# 20. Survived ~ Age + Sex_male + Pclass + Fare + Title_Miss +
#                Title_Mr + Title_Mrs + Embarked_Q +  Embarked_S + 
#                FamilySize + FamilyID2_Andersson_7 + 
#                FamilyID2_Asplund_7 + FamilyID2_Baclini_4 +
#                FamilyID2_Becker_4 + FamilyID2_Carter_4 + 
#                FamilyID2_Dean_4 + FamilyID2_Ford_5 + 
#                FamilyID2_Fortune_6 + FamilyID2_Goodwin_8 + 
#                FamilyID2_Herman_4 + FamilyID2_Johnston_4 + 
#                FamilyID2_Laroche_4 + FamilyID2_Lefebre_5 + 
#                FamilyID2_Palsson_5 + FamilyID2_Panula_6 + 
#                FamilyID2_Rice_6 + FamilyID2_Ryerson_5 + 
#                FamilyID2_Sage_11 + FamilyID2_Skoog_6 + 
#                FamilyID2_Small + FamilyID2_West_4 + SibSp +                                      
#                Parch                                              0.8380 (+/- 0.02)    0.78947



# The best prediction for survival with SVM was achieved using the 6th model.


#
#
## We could also tune the SVM model to find the optimal parameter values.
## We will try using all available predictors,
#predictor=list(train_set_new)
#predictors_to_remove = ['Survived', 'Cabin', 'Name', 'Surname', 'Ticket', 'FamilyID', 'Perished', 'PassengerId']
#predictor= list(set(predictor).difference(set(predictors_to_remove)))
#x_train = train_set_new.loc[:,predictor]
#y_train = train_set_new.loc[:,'Survived']
#x_test = test_set_new.loc[:,predictor]
#    
# 
## Parameters that will be tuned  
#
## kernel: represents the type of hyperplane used to separate the data. 
## 'rbf' and 'poly' uses a non linear hyper-plane
#kernel = ['rbf', 'poly']
## gamma: is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly 
## fit the training data set
#gamma = [0.01, 0.1, 1, 10, 100, 'scale']
##C: is the penalty parameter of the error term. It controls the trade off between 
## smooth decision boundary and classifying the training points correctly.
#C = [0.01, 0.1, 1, 10, 100, 1000]
##degree: Degree of the polynomial kernel function ('poly').Ignored by all other kernels.
#degree = [3,4,10]
#
#
#svm = SVC(random_state=1)
#
#param_grid = {'kernel': kernel, 'gamma': gamma, 'C': C, 'degree': degree}
#
#
#svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=150, n_jobs=n_jobs)
## Set the verbose parameter in GridSearchCV to a positive number 
## (the greater the number the more detail you will get) to print out progress
#svm_cv.fit(x_train, y_train)
#
#
#print("Best parameters found on training set:")
#print(svm_cv.best_params_)
##Best parameters found on training set:
##
#print('Best score:',svm_cv.best_score_)
## Best score: 
#
#
#svm_optimised = svm_cv.best_estimator_
#svm_optimised.fit(x_train, y_train)
#
#
## Performance on the training set
#
## Compute the accuracy using the shuffle validator
#test_classifier(svm_optimised)
##Model 1 - Accuracy on Training Set: 
#
#
## Performance on the testing set
#y_pred = svm_optimised.predict(x_test).astype(int)
## Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
#my_solution  = pd.DataFrame({
#        "PassengerId": test_set_new.loc[:,'PassengerId'],
#        "Survived": y_pred
#    })
#my_solution.to_csv('svm_optimised.csv', index=False)
## Kaggle score 




#####   Voting Classifier
    
# A "Voting" classifier can be used to apply multiple conceptually divergent classification models 
# to the same data set and will return the majority vote from all of the classifiers. 
    
    
import sklearn.ensemble as ske    

predictors = ['Age','Sex_male','Pclass','Fare','SibSp','Parch','Embarked_Q','Embarked_S', 
                'Title_Miss','Title_Mr','Title_Mrs','FamilySizeNote_Singleton',
                'FamilySizeNote_Small_Family','TicketCount']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Survived']
x_test = test_set_new.loc[:,predictors]



# To create the Extremely Randomized trees, we first initialize an instance of an untrained Extremely 
# Randomized trees classifier
extra_tree = ExtraTreesClassifier(n_estimators=100, max_depth=10, max_features='sqrt', 
                                       min_samples_split=2, min_samples_leaf=3, random_state=1)

# To create the Logistic Regression, we first initialize an instance of an 
# untrained Logistic Regression classifier
log_reg = LogisticRegression(solver ='lbfgs', random_state=1) # penalty = 'l2', C = 1

# To create the SVM Classifier, we first initialize an instance of an 
# untrained SVM Classifier
svm = SVC(C=1, gamma='auto', kernel='rbf', # default values
           random_state=1) 


voting = ske.VotingClassifier([('ExTree', extra_tree), ('LogReg', log_reg), ('SVM', svm)])


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
voting.fit(x_train, y_train)

    

# Performance on the training set

# Compute the accuracy using the shuffle validator
test_classifier(voting)
#Model 1 - Accuracy on Training Set: 0.8416 (+/- 0.02)


# Performance on the testing set

y_pred = voting.predict(x_test).astype(int)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
my_solution  = pd.DataFrame({
        "PassengerId": test_set_new.loc[:,'PassengerId'],
        "Survived": y_pred
    })
my_solution.to_csv('voting.csv', index=False)
# Kaggle score 0.79425




# From the experimental section, we see that the most accurate model is the Extra-Tree 
# when using the predictor variables Age, Sex_male, Pclass, Fare, SibSp, Parch, Embarked_Q,
# Embarked_S, Title_Miss, Title_Mr, Title_Mrs, FamilySizeNote_Singleton, FamilySizeNote_Small_Family
# and TicketCount. The accuracy of this model on the training set is 0.8388 (+/- 0.02) and on the 
# test set 0.80861. 
    