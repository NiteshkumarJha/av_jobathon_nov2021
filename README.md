# Analytics vidhya JOB-A-THON - Nov 2021

## Introduction
The objective of this JOB-A-THON is to predict which employee is most likely to quit in next 6 months based on employee's details and its monthly performance features. Initial dataset contains 19,104 rows and 13 columns and task was to predict survival for 741 employee present in test dataset. This problem can be addressed by survival analysis based on particular use case and datasets available. To solve this, below steps has been adopted:

 1. **Initial EDA :** Explore initial dataset and decide data preprocessing and modeling strategy.
 2. **Data preprocessing :** Prepare data in specific format for performing survival analysis.
 3. **EDA :** Peform Exploratory data analysis to understand relationship between target and features.
 4. **Modeling :** Explore various models and identify best model.
 5. **Prediciton :** Create prediction for test data and prepare submission file.

## Programming language and packages

 + **Programming language :** `python` 
 + **packages :**`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lifelines`
 + **notebook :** `google colab`

## Data preprocessing

Survival analysis require data in particular format. Below steps are peformed to create target and features during this analysis:

### Target variable

Two target variable has been created :

 + **time** : Employee stay time(in months). 
 + **event** : Whether employee left or not. 

Since data is only given till Dec'2017, we can conside this as an example of censored dataset, mainly right censored dataset where employee may have left after study period.

Below `python` code is used to generate target dataset.

```
#####################################################################################
#                          Creating target set                                      #
#####################################################################################

###### changing data type for date variables
train['MMM-YY'] = pd.to_datetime(train['MMM-YY'])
train['Dateofjoining'] = pd.to_datetime(train['Dateofjoining'])
train['LastWorkingDate'] = pd.to_datetime(train['LastWorkingDate'])

###### creating target variable for survival analysis
# date of joining table
date_of_joining = train[["Emp_ID", "Dateofjoining"]].drop_duplicates()
# last working date table
last_working_date = train[["Emp_ID", "LastWorkingDate"]].dropna().drop_duplicates()

##### target table
target = pd.merge(date_of_joining, last_working_date, on = "Emp_ID", how = "left")

# creating event column
target["event"] = np.where(target["LastWorkingDate"].isnull()==True, False, True)

# creating time for event column
target['time'] = \
np.where(target["LastWorkingDate"].isnull()==True, (pd.to_datetime("2017-12-31") - target['Dateofjoining']), 
         (target['LastWorkingDate'] - target['Dateofjoining']))

target['time'] = target['time'] / np.timedelta64(1, 'M')
target['time'] = target['time'].round(0).astype(int)

# keeping only revalant column
target = target[["Emp_ID", "time", "event"]]

```

### Feature set :

Given dataset contains monthly observation of each employee. For conducting survival analysis, this dataset is aggregated at individual employee level. Below features has been created:

  1. Gender (Binary) : Gender of Employee. Convered this feature in binary format.(1/0)
  2. City (Categorical) : City of Emoloyee. Performed Dummy Encoding using `pandas` in-built dummy encoder. After EDA, this feature is dropped from modeling due to high cardinality.
  3. Education_Level (Categorical) : Categorical feature. Performed Dummy Encoding(One-hot Encoding).
  4. Age (Continous) : Max Age of employee.
  5. Avg_Salary (Continous): Categorical feature. Performed Dummy Encoding(One-hot Encoding).
  6. Salary_Increase (Binary) : Binary feature. Recoded into 1 and 0.
  7. Designation (Ordinal) : Highest designation hold by employee.
  8. Promotion (Ordinal) : Level of promotion
  9. Total Business Value (Continous): Average business value
  10. Median_quarterly_rating (Ordinal) : Median quarterly rating.

Below `python` code is used to generate target dataset.

```
#####################################################################################
#                          Creating features set                                    #
#####################################################################################
# gender_city_edu_age_features
gender_city_edu_age_features = pd.pivot_table(train, 
                          index = ["Emp_ID", "Gender", "City", "Education_Level"],
                          values = ["Age"],
                          aggfunc = np.max).reset_index()

# salary features
salary_features = pd.pivot_table(train, 
                          index = ["Emp_ID"],
                          values = ["Salary"],
                          aggfunc = [np.mean, np.std]).reset_index()
salary_features.columns = ["Emp_ID", "Avg_Salary", "Salary_Std"]
salary_features["Salary_Increase"] = np.where(salary_features["Salary_Std"]>0,1,0)
salary_features = salary_features[["Emp_ID", "Avg_Salary", "Salary_Increase"]]

# Designation feature
train["Promotion"] = train["Designation"] - train["Joining Designation"]
designation = pd.pivot_table(train, 
                             index = ["Emp_ID"],
                             values = ["Designation", "Promotion"],
                             aggfunc = np.max).reset_index()
designation.columns = ["Emp_ID", "Designation", "Promotion"]

# total business value feature
tbv = pd.pivot_table(train, 
                     index = ["Emp_ID"],
                     values = ["Total Business Value"],
                     aggfunc = np.mean).reset_index()

# Quarterly Rating
quart_rating = pd.pivot_table(train, 
                              index = ["Emp_ID"],
                              values = ["Quarterly Rating"],
                              aggfunc = [np.median]).reset_index()
quart_rating.columns = ["Emp_ID", "Median_quarterly_rating"]

# creating final features columns
features_set = \
gender_city_edu_age_features.merge(salary_features, on = "Emp_ID", how = "left")\
.merge(designation, on = "Emp_ID", how = "left")\
.merge(tbv, on = "Emp_ID", how = "left")\
.merge(quart_rating, on = "Emp_ID", how = "left")

##### feature transformation
# binary encoding - gender 
features_set["Gender"] = np.where(features_set["Gender"] == "Male", 1, 0)

# dummy encoding - City, Education_Level
features_set = \
pd.concat([features_set.drop(["City", "Education_Level"],axis = 1), 
           pd.get_dummies(features_set[["City", "Education_Level"]], drop_first = True)], axis = 1)
```

## Exploratory data analysis

Detailed EDA has been conducted on this dataset. Below are the some major findings are :

 + Employee attrition rate is high approx 68%.
 + Time to stay is following skewed distribution with mean stay is ~14 months whereas median stay is 6 months. It is right skewed with heavy tail influenced by presence of high stay by some employees.
 + Total Business value, Age and Average salary are showing impact on leaving and stay time. Aged employee, Employee with high average salary employee who brings high business value do not prefer to leave.
 + Rest factor can not be concluded having any significant impact on employee leaving the organisation.
 + City is having high cardinality with creating high sparsity (high number of 0) in dataset. This feature can be dropped from modeling. 
 
## Model training

At first, historical training dataset is devided into two parts : train and val. This splitting is done based on randomness. For this, `scikit-learn` package has been used. Below code has been used to crete training and validation dataset.

```
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(training_dataset, test_size = 0.3, random_state = 123)

print("Shape of training data :", train_df.shape)
print("Shape of validation data :", val_df.shape)

# Shape of training data : (1666, 41)
# Shape of validation data : (715, 41)
```
Below model has been explored for performing survival regression from `lifelines` package.

 1. **Semi-parametric models**
   + Cox’s proportional hazard model
 2. **Parametric models**
   + Weibull Accerlated Failure time model
   + Log Logistic Accerlated Failure time model
   + Log Normal Accerlated Failure time	model
 
Each model has been trained on train dataset and cross validated on validation dataset. For checking model performance, concordance index and log likelihood has been used as model metric. 

The **concordance index** or **C-index** is a generalization of the area under the ROC curve (AUC) that can take into account censored data. It represents the global assessment of the model discrimination power: this is the model’s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores. Similarly to the AUC, C-index = 1 corresponds to the best model prediction, and C-index = 0.5 represents a random prediction. Below is the peformance of each models:

|Model Name|C-index(val_concordance_index)| Score | Private Score | Public Leaderboard | Private Leaderboard |
|:--------:|:----------------------------:|:-----:|:-------------:|:------------------:|:-------------------:|
|Cox Proptional Hazard	| 0.84 | 0.70387630275938 | 0.719531378271163 | | |
|Weibull Accerlated Failure time| 0.85 | 0.714445290682914 | 0.705056179775281 | 39 | 58 |
|Log Logistic Accerlated Failure time| 0.83 | 0.696113782051282 | 0.71388022017658| | |
|Log Normal Accerlated Failure time| 0.82 | Not submiited | Not submiited | | |

> As per validation dataset and Score, Weibull Accerlated Failure time modle is perfoming better and hence selected as final submisison model. However, it seems Cox Proptional Hazard model worked better on Private Score. Below is the code used for fitting both models.

```
from lifelines import CoxPHFitter

# dropping some features
feature_set = ['Gender', 'Age', 'Avg_Salary',
               'Salary_Increase', 'Designation', 'Promotion', 'Total Business Value',
               'Median_quarterly_rating', 'Education_Level_College', 
               'Education_Level_Master']

train_df = train_df[feature_set + target_cols]

# fitting cox regression model
cph = CoxPHFitter()
cph.fit(train_df, duration_col='time', event_col='event')
cph.print_summary()

# plotting variables
plt.figure(figsize = (15,5))
cph.plot();

# checking assumptions
cph.check_assumptions(train_df, show_plots = True)

# Weibull Accerlated Failure time model
from lifelines import WeibullAFTFitter

# imputing 0 with small quantities
train_df_v2 = train_df.copy()
train_df_v2["time"] = np.where(train_df_v2["time"] == 0, 0.01, train_df_v2["time"])

aft = WeibullAFTFitter()
aft.fit(train_df_v2, duration_col='time', event_col='event', ancillary=True)

aft.print_summary()

plt.figure(figsize = (20,7))
aft.plot();

### 
```
For predicting on employee in test dataset,  survival probability for next 6 month and median survival time is used. For this, a utility function has been created and final submission file has been generated.

```
# function for generating predicitons
def predict_survival(dataset, test_df, event_var, time_var, model, period, ID):
  
  # creating censored subjects data
  censored_subjects = dataset.loc[~dataset[event_var].astype(bool)]
  censored_subjects_last_obs = censored_subjects[time_var]

  # predicting survival probability and median survival period
  pred_df_cph = \
  model.predict_survival_function(censored_subjects, 
                                  conditional_after=censored_subjects_last_obs, 
                                  times = [period]).transpose()
  pred_df_cph["Median Remianing Survival"] = \
  model.predict_median(censored_subjects, conditional_after=censored_subjects_last_obs)
  pred_df_cph.columns = ["Survival Probability", "Median_Remianing_Survival"]
  pred_df_cph["Target"] = np.where(pred_df_cph["Median_Remianing_Survival"]<=period,1,0)
  pred_df_cph["Target"] = pred_df_cph["Target"].astype(int)
  output_df = dataset[[ID]]
  output_df = output_df.join(pred_df_cph)
  output_df = output_df.dropna()
  output_df = pd.merge(test_df, output_df, on = ID, how = "left")
  
  return output_df
  
 # generating predictions 
output_df_waft = \
predict_survival(training_dataset, test, 
                 event_var = 'event', time_var = 'time', 
                 model = aft, period = 6, ID = "Emp_ID")

## saving output file
output_df_waft.to_csv("/content/drive/MyDrive/Data Science/survival_analysis/output/submission_waft.csv", 
                      index = False)
```

Some of models did not get explored due to limited amount of time as well as these are little bit of advanced and require some hands-on knowleadge on handling of another package `scikit-survival`, `Pysurvival` and others.

 **Non-linear models**
   + Survival Tree `scikit-survival`
   + Random Survival forest `scikit-survival`
   + Extra Survival Trees `scikit-survival`
   + GradientBoostingSurvivalAnalysis `scikit-survival`
   + Survival Support Vector Machine `scikit-survival`
   + Conditional Survival Forest model `Pysurvival`
   + Linear Multi-Task Logistic Regression `Pysurvival`
   + Neural Multi-Task Logistic Regression `Pysurvival`


## Next steps

Exploring feature selections techniques, regularization and other models can improve model quality.


