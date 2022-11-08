import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import json

pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(precision=4, suppress=True)

# Read in input file
file = 'glo_sample.csv'
flight_df = pd.read_csv(file)

# Copying only relevant columns of data
xy_df = flight_df[['DepartureDateTimeUTC', 'ArrivalDateTimeUTC', 'DepartureCountry', 'price']].copy()

# Converting to datetime object
xy_df['DepartureDateTimeUTC'] = pd.to_datetime(xy_df['DepartureDateTimeUTC'])
xy_df['ArrivalDateTimeUTC'] = pd.to_datetime(xy_df['ArrivalDateTimeUTC'])

# Calculating flight duration
xy_df['Duration'] = (xy_df['ArrivalDateTimeUTC'] - xy_df['DepartureDateTimeUTC']).astype('timedelta64[m]')/60.0

# Calculating price per hour
xy_df['PricePerHour'] = xy_df['price']/xy_df['Duration']

# Converting date to day of week
xy_df['Day_of_Week'] = xy_df['DepartureDateTimeUTC'].dt.dayofweek

# Removing "inf" and "NaN" samples
xy_df.replace([np.inf, -np.inf], np.nan, inplace=True)
xy_df.dropna(inplace=True)

# Creating a dictionary of country codes
PYTHONHASHSEED = 10
country_codes_dict = {}
countries = list(set(xy_df['DepartureCountry']))
for i in range(len(countries)):
    country_codes_dict[countries[i]] = i

print(json.dumps(country_codes_dict))

# Mapping country names to country codes
xy_df['DepartureCountryCode'] = xy_df['DepartureCountry'].map(country_codes_dict)
xy_df['DepartureDate'] = xy_df['DepartureDateTimeUTC'].dt.date

# Sorting the data by departure dates
xy_df.sort_values(by=['DepartureDate', 'DepartureCountry'], inplace=True)

# Adding a column of ones to setup for counting the number of flights
xy_df['Ones'] = 1
xy_grp_df = pd.DataFrame()

# Grouping the data by departure dates at a frequency of one day
# Price per hour is averaged over the cost of all flights originating from one country on that particular day
# Number of flights originating from each country on that particular day is determined
xy_grp_df['DepartureCountryCode'] = xy_df.groupby([pd.Grouper(key='DepartureDateTimeUTC', freq='D'),
                                                   pd.Grouper(key='DepartureCountryCode')])['DepartureCountryCode'].mean().to_frame().astype(int)
xy_grp_df['PricePerHour'] = xy_df.groupby([pd.Grouper(key='DepartureDateTimeUTC', freq='D'),
                                           pd.Grouper(key='DepartureCountryCode')])['PricePerHour'].mean().to_frame()
xy_grp_df['Day_of_Week'] = xy_df.groupby([pd.Grouper(key='DepartureDateTimeUTC', freq='D'),
                                          pd.Grouper(key='DepartureCountryCode')])['Day_of_Week'].mean().to_frame()['Day_of_Week'].astype(int)
xy_grp_df['No_of_Flights'] = xy_df.groupby([pd.Grouper(key='DepartureDateTimeUTC', freq='D'),
                                            pd.Grouper(key='DepartureCountryCode')])['Ones'].sum().to_frame()

# One-hot encoding country codes
ohe = OneHotEncoder()
X_Country_Code = ohe.fit_transform(xy_grp_df.DepartureCountryCode.values.reshape(-1,1)).toarray()

# Converting dataframe to numpy array (leaving out the country code)
X = xy_grp_df[xy_grp_df.columns[1:-1]].to_numpy()
y = xy_grp_df[xy_grp_df.columns[-1]].to_numpy()

# Applying MinMaxScaler transform to the "price per hour" and the "day of the week" columns
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

# Inserting the one-hot-encoded country code array
X_scaled = np.insert(X_scaled, [0], X_Country_Code, axis=1)

print(X_scaled)
print(X_scaled.shape, y.shape)
print(y)

# Test-train split
rand_seed = random.randint(0, 100000000)
X_tr, X_te, y_tr, y_te  = train_test_split(X_scaled, y, test_size=0.2, random_state=rand_seed)

# Random Forest Regression with grid-search on the n_estimators hyper-parameter
max_CVS_mean = 0.0
min_CVS_std = np.infty
for i in range(1, 50):
    reg1 = RandomForestRegressor(n_estimators=i)
    scores = cross_val_score(reg1, X_tr, y_tr, cv=5)
    if scores.mean() > max_CVS_mean:
        max_CVS_mean = scores.mean()
        max_CVS_mean_std = scores.std()
        max_CVS_mean_i = i
    if scores.std() < min_CVS_std:
        min_CVS_std = scores.std()
        min_CVS_std_mean = scores.mean()
        min_CVS_std_i = i

# Best model (using Random Forest Regression) as far as maximizing the mean of the cross-validation R2 score is concerned
print(f'Random Forest Regression Max Mean Model: n_estimators = {max_CVS_mean_i}, CVS mean = {max_CVS_mean}, CVS std: {max_CVS_mean_std}')

# Best model (using Random Forest Regression) as far as minimizing the std of the cross-validation R2 score is concerned
print(f'Random Forest Regression Min Std Model: n_estimators = {min_CVS_std_i}, CVS mean = {min_CVS_std_mean}, CVS std: {min_CVS_std}')

# Applying the best Random Forest Regression model to prediction on the test data
reg1 = RandomForestRegressor(n_estimators=max_CVS_mean_i)
reg1.fit(X_tr, y_tr)
y_te_pred = reg1.predict(X_te)
print(f'Random Forest Regression with the Tuned Max Mean Model with n_estimators = {max_CVS_mean_i}: '
      f'R2 score on test data = {r2(y_te, y_te_pred)}')


