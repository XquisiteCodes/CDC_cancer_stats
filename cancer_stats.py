# %%
#Importing python modules for data importaion, analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dataset = pd.read_csv('/home/ezenwa/Documents/Uploaded to Git/CDC_Cancer_stats/BYAGE.TXT', sep = '|', low_memory = False)

# %%

pd.set_option('display.max_columns', 15)
dataset.head()

# %%
print(dataset.nunique())
print(dataset.info(memory_usage = 'deep'))


# %%
# Next I want see the percentage of people that died from the cancer and the percentage that were recorded to have cancer but did not pass
dataset_event_type = dataset.groupby('EVENT_TYPE').size()
dataset_event_type

# %%
sns.countplot(x = 'EVENT_TYPE', data = dataset)
# I am going to be taking 'EVENT_TYPE' feature as my target feature for the putpose of this analysis, so I want to further investigate features that determine the mrtality or a manageable incident

# %%
dataset.groupby(['EVENT_TYPE','SEX']).size()

# Grouping dataset by EVENT_TYPE and SEX, we can see more incidents of females with cancer and also more females passed from cancer.

# %%
# it seems somes numbers have been imputed as strings, I will convert them to numbers using pandas eval method
def eval(a):
    try:
        return pd.eval(a)
    except SyntaxError:
        return np.nan
dataset.COUNT = dataset.COUNT.apply(eval)

# %%
dataset.CI_LOWER = dataset.CI_LOWER.apply(eval)

# %%
dataset.CI_UPPER = dataset.CI_UPPER.apply(eval)

#%%
dataset.RATE = dataset.RATE.apply(eval)
dataset.info()

#%%
dataset[pd.isnull(dataset.RATE)].head(10)
#it seems NaN values in RATE occur exactly in the same location accross the other columns containing NaNs

#%%
#I will be dropping rows that contain NaNs since they occur simmultaneously accross 4 columns
dataset.dropna(inplace = True)

#%%
dataset.info()
#Great our data is clean, now we can proceed to analysis and EDA
#%%
dataset_age = dataset.groupby(['EVENT_TYPE','AGE'])['COUNT'].sum().reset_index()
dataset_age

#%%
plt.figure(figsize = (15,8))
sns.barplot(x = 'AGE', y = 'COUNT', hue = 'EVENT_TYPE', data = dataset)

#As expected incidence of cancer is higher from age 50 and mortality is even higher the older you get.
#%%
dataset_race = dataset.groupby(['RACE', 'EVENT_TYPE']).size().reset_index()
dataset_race.columns = ['RACE', 'EVENT_TYPE', 'SIZE']
dataset_race
#%%
plt.figure(figsize=(15, 8))
sns.barplot(x = 'RACE', y = 'SIZE', hue = 'EVENT_TYPE', data = dataset_race)
#race seems to have effect on incidence of cancer

#%%
dataset_site = dataset.groupby(['SITE','EVENT_TYPE']).size().reset_index()
dataset_site.columns = ['SITE', 'EVENT_TYPE', 'SIZE']
dataset_site

#%%
plt.figure(figsize=(50, 10))
sns.barplot(x = 'SITE', y = 'SIZE', hue = 'EVENT_TYPE', data = dataset_site)


#%%
#incidence here refers to reported cases of cancer that didn't result in mortality. I want generate a model to predict if a reported case of cancer will most likely result in mortality or not. I will be using logistic regression for this modelling

#%%
#First  will Encode categoriacal data using sklearn's label encoder
from sklearn.preprocessing import LabelEncoder
#dropping irrelevant columns
dataset_modeling = dataset.drop(['POPULATION','RATE','YEAR','COUNT'], axis = 1)
cat_data = ['AGE','EVENT_TYPE','RACE','SEX','SITE']
#%%
encode = LabelEncoder()
dataset_modeling[cat_data] = dataset_modeling[cat_data].apply(encode.fit_transform)
dataset_modeling.head()


#%%
y = dataset_modeling.EVENT_TYPE
X = dataset_modeling.drop(['EVENT_TYPE'], axis = 1)
print(y.head())
print(X.head())
#%%
#Now I will standardize the training dataset
from sklearn.preprocessing import StandardScaler
#getting columns first
cols = X.columns
standard = StandardScaler()
scaled_X = standard.fit_transform(X)

#%%
#converting scaled_X back to a dataframe
scaled_X = pd.DataFrame(scaled_X, columns=cols)
scaled_X.head()

#now our data is normalized
#%%
from sklearn.linear_model import LogisticRegression  #for training and fitting the model
from sklearn.model_selection import cross_val_score  #for cross validation of dataset
LR_model =LogisticRegression()

#%%
scores = cross_val_score(LR_model, scaled_X,y, scoring = 'precision_micro')
scores1 = cross_val_score(LR_model, scaled_X,y, scoring = 'recall_micro')
scores2 = cross_val_score(LR_model, scaled_X,y, scoring = 'accuracy')
scores3 = cross_val_score(LR_model, scaled_X,y, scoring = 'neg_mean_absolute_error')
#%%
print(scores)
print(scores1)
print(scores2)
print(scores3)
#%%
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(scaled_X, y, test_size = 0.3)
#%%
LR_2 = LogisticRegression()
model_2 = LR_2.fit(train_X, train_y)
#%%
from sklearn.metrics import mean_squared_error
mean_squared_error(test_y, model_2.predict(test_X))

#%%


#%%


#%%


#%%

#%%


#%%