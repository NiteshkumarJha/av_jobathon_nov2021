# Analytics vidhya JOB-A-THON - Nov 2021

## Introduction
The objective of this JOB-A-THON is to predict which employee is most likely to quit in next 6 months based on employee's details and its monthly performance features. Initial dataset contains 19,104 rows and 13 columns and task was to predict survival for 741 employee present in test dataset. This problem can be addressed by survival analysis based on particular use case and datasets available. To solve this, below steps has been adopted:

 1. **Initial EDA :** Explore initial dataset and decide data preprocessing and modeling strategy.
 2. **Data preprocessing :** Prepare data in specific format for performing survival analysis.
 3. **EDA :** Peform Exploratory data analysis to understand relationship between target and features.
 4. **Modeling :** Explore various models and identify best model.
 5. **Prediciton :** Create prediction for test data and prepare submission file.

## Data preprocessing

Survival analysis require data in particular format. Below steps are peformed to create target and features during this analysis:

### Target variable

Two target variable has been created :

 + **time** : Employee stay time(in months). 
 + **event** : Whether employee left or not. 

Since data is only given till Dec'2017, we can conside this as an example of censored dataset, mainly right censored dataset where employee may have left after study period.

Below python code is used to generate target dataset.

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

Below are the feature information and respective transformation used during modeling:

 + Target : Sales (Since sales variable is skewed, **sqrt** transformation is performed for training purpose.)
 + Features :
     
     1. Store_id : Indicator variable. Used as is in the model. In addition to Store_id, one additional feature is also created based on target encoding. This feature is created by taking average of sqrt(Sales) by each store_id. This feature is merged with train and test data based on Store_id. This features help us to understand size of each store.
     2. Discount : Binary feature. Recoded into 1 and 0.
     3. Store_Type : Categorical feature. Performed Dummy Encoding(One-hot Encoding).
     4. Location_Type : Categorical feature. Performed Dummy Encoding(One-hot Encoding).
     5. Region_Code : Categorical feature. Performed Dummy Encoding(One-hot Encoding).
     6. Holiday : Binary feature. Recoded into 1 and 0.
     7. Date : Date indicator. Based on this, new features has been created to understand time related pattern. These are Year, Month, Day, Week_of_year, Day_of_week, Weekend, Month_Start, Month_End, Quarter_Start, Quarter_End, Year_Start, Year_End. This date features are used as is in the model except Day_of_week feature. It is converted using One-hot encoding. 

## Exploratory data analysis

Detailed EDA has been conducted on this dataset. Below are the some major findings are :

 + Sales is not following normal distribution. It is right skewed with heavy tail influenced by presence of high sales in some stores or day.
 + Zero Sales were recorded in some instances.
 + Discount, Holiday and Store_Type are having major impact on Sales.
 + High Sales were observed during Weekend compare to weekdays.
 + There are fluctuations in store level average sales. Some stores are showing abnormally high sales.
 
## Model training

At first, historical training dataset is devided into two parts : train and val. This splitting is done based on time driven split. Last two month dataset kept for validation and rest all are kept as training set. Various model has been explored and tested using this dataset. Some of these are :

 1. Random Forest model
 2. XGBoost model
 3. Prophet model by individual stores.

> **Random Forest** model is found as best performing model. This model is again retrained on whole training set with below parameters:

**Target** : $ \sqrt{Sales} $

**Features** : Store_id, Holiday, Discount, Year, Month, Day, Week_of_year, Weekend, Month_Start, Month_End, Quarter_Start,
               Quarter_End, Year_Start, Year_End, Store_Type_S1, Store_Type_S2, Store_Type_S3, Store_Type_S4, Location_Type_L1,
               Location_Type_L2, Location_Type_L3, Location_Type_L4, Location_Type_L5, Region_Code_R1, Region_Code_R2,
               Region_Code_R3, Region_Code_R4, Day_of_week_0, Day_of_week_1, Day_of_week_2, Day_of_week_3, Day_of_week_4, Day_of_week_5,
               Day_of_week_6, Avg_Sales_Sqrt
               
 **Hyperparameters**  :      
   + n_estimators : 500
   + min_samples_leaf : 3
   + max_features : sqrt
   + n_jobs : -1
   + random_state: 123
 
## Next steps

Exploring additional features like lag features, rolling average features can improve model quality. In addition to this. Neural network algorithms like LSTM can also improve model quality.
