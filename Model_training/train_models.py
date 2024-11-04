
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import uniform



# Set options to show all columns
pd.set_option('display.max_columns', None)

# Load .pkl file and generate pandas dataframe
input_pkl = r'..\data\clean\after_step_3b_outliers_cat.pkl' # Fill your path to file
df_1 = pd.read_pickle(input_pkl)

df_1_shape = df_1.shape # Pass the shape of the dataframe to a variable for summary at end of the outlier detection and removal part
df_1.info()

# **This selection is made based on the correlation matrix in Team_6_Step_4**

selected_columns = ['Price','Number_of_bedrooms','Living_area','Number_of_facades','State_of_building','epc','landSurface','Has_Assigned_City','Province'] 
df = df_1[selected_columns].copy(deep=True)

df.info()

df.head(30)

# **Checking for missing values**

df.isnull().sum()

# **Dealing with categorical Features**

categorical_df = df.select_dtypes(include=['category','object'])

categorical_df.info()

# **Encoding Province** - get_dummies

province_dummies= pd.get_dummies(categorical_df['Province'], drop_first=True)
province_dummies.head()

df = pd.concat([df,province_dummies], axis=1)
df= df.drop('Province', axis=1)

# **Encoding EPC** - Ordinalencoder
categorical_df['epc'].value_counts()

list_epc = categorical_df['epc'].values.tolist()
unique_epc = list(set(list_epc))
unique_epc.sort(reverse=True)

print(unique_epc)

epc_val = categorical_df[['epc']].values

encoder = OrdinalEncoder(categories=[unique_epc])

df['Encoded_epc'] = encoder.fit_transform(epc_val)
df= df.drop('epc', axis=1)



# **Encoding State_of_building** - Ordinalencoder

categorical_df['State_of_building'].value_counts()

list_state = categorical_df['State_of_building'].values.tolist()
unique_state = list(set(list_state))
print(unique_state)

sort_unique_state = ['To renovate','To be done up','Good', 'Just renovated','As new']

state_val = categorical_df[['State_of_building']].values

encoder = OrdinalEncoder(categories=[sort_unique_state])

df['Encoded_state_of_building'] = encoder.fit_transform(state_val)
df= df.drop('State_of_building', axis=1)

df['Encoded_state_of_building'].value_counts()

df.info()

df.isnull().sum()

# **Splitting the dataset**

X = df.drop('Price', axis = 1)
y = df['Price'].values
print(type(X), type(y))

print(X.shape)
print(y.shape)

# **Splitting in train and test data**


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

reg_all.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# **CENTRING AND SCALING**

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train), np.std(X_train))

# **REGRESSION MODELS**

# **Linear Regression**

reg = LinearRegression()

# *Cross-validation*

#cross-validation peformance
kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_results = cross_val_score(reg, X_train, y_train, cv=kf)
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results,[0.025,0.075]))

# *Train the Model on the Training Set*

reg.fit(X_train, y_train)


reg.score(X_train,y_train)

# *Evaluate Model on the Test Set*

y_pred = reg.predict(X_test)
reg.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

feature_weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': reg.coef_
})

print(feature_weights)

ridge = Ridge(alpha=1.0)

kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_results = cross_val_score(ridge, X_train, y_train, cv=kf)
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results,[0.025,0.075]))

ridge.fit(X_train, y_train)


ridge.score(X_train,y_train)

# *Evaluate Model on the Test Set*

y_pred = ridge.predict(X_test)
ridge.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# *Hyper parameter tuning*

param_grid = {'alpha': np.linspace(0.0001,1,10), 'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']}
rscv = RandomizedSearchCV(ridge, param_grid, cv = kf, n_iter=2)
rscv.fit(X_train, y_train)
print("Best Parameters:", rscv.best_params_)
print("Best Cross-Validation Score:", rscv.best_score_)

best_model = rscv.best_estimator_


best_model.score(X_train,y_train)

y_pred = best_model.predict(X_test)
best_model.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

feature_weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': best_model.coef_
})


print(feature_weights)

# **Lasso**

lasso = Lasso(alpha=1)

# *Cross-validation*

#cross-validation peformance
kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_results = cross_val_score(lasso, X_train, y_train, cv=kf)
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results,[0.025,0.075]))
# *Train the Model on the Training Set*

lasso.fit(X_train, y_train)

lasso.score(X_train,y_train)

# *Evaluate Model on the Test Set*

y_pred = lasso.predict(X_test)
lasso.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# *Hyper parameter tuning*

param_grid = {'alpha': np.linspace(0.0001, 1, 10)}
rscv = RandomizedSearchCV(lasso, param_distributions=param_grid, cv=kf, n_iter=10, random_state=42)
rscv.fit(X_train, y_train)
print("Best Parameters:", rscv.best_params_)
print("Best Cross-Validation Score:", rscv.best_score_)

best_model = rscv.best_estimator_


best_model.score(X_train,y_train)

y_pred = best_model.predict(X_test)
best_model.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

feature_weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': best_model.coef_
})


print(feature_weights)

# **ELASTICNET**

elas = ElasticNet(alpha=1)

kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_results = cross_val_score(elas, X_train, y_train, cv=kf)
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results,[0.025,0.075]))

elas.fit(X_train, y_train)

elas.score(X_train,y_train)

y_pred = elas.predict(X_test)
elas.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

param_distributions = {'alpha': uniform(0.01, 10),'l1_ratio': uniform(0, 1)}
rscv = RandomizedSearchCV(lasso, param_distributions=param_grid, cv=kf, n_iter=10, random_state=42)
rscv.fit(X_train, y_train)
print("Best Parameters:", rscv.best_params_)
print("Best Cross-Validation Score:", rscv.best_score_)

best_model = rscv.best_estimator_

best_model.score(X_train,y_train)

y_pred = best_model.predict(X_test)
best_model.score(X_test,y_test)

print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

feature_weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': best_model.coef_
})


print(feature_weights)

# **RANDOM FORREST**

reg_rf = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)


#cross-validation peformance
kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_results = cross_val_score(reg_rf, X_train, y_train, cv=kf)
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results,[0.025,0.075]))

reg_rf.fit(X_train, y_train)

# Access the OOB Score
oob_score = reg_rf.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Making predictions on the same data or new data
y_pred = reg_rf.predict(X_test)

# Evaluating the model
mse = root_mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
# Save data to new csv file
output_csv = r'..\data\clean\model_training.csv'  # Fill your path to file
df.to_csv(output_csv, index=False)
# Save data to new pkl file
output_pkl = r'..\data\clean\model_training.pkl' # Fill your path to file
with open(output_pkl, 'wb') as f:
    pickle.dump(df, f)


