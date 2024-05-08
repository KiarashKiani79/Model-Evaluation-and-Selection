from my_common_imports import *
from loads_and_split_data import LoadAndSplitData

# Load & split the data 
x_train, y_train, x_cv, y_cv, x_test, y_test = LoadAndSplitData() 

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
x_cv_mapped = poly.transform(x_cv)

# Feature scaling
poly_scalar = StandardScaler()
x_train_mapped_scaled = poly_scalar.fit_transform(x_train_mapped)
x_cv_mapped_scaled = poly_scalar.transform(x_cv_mapped)

# Train the model
model = LinearRegression()
model.fit(x_train_mapped_scaled, y_train)

# Compute MSE for training set & cross validation set
yhat_xtrain = model.predict(x_train_mapped_scaled)
yhat_xcv = model.predict(x_cv_mapped_scaled)

training_MSE = mean_squared_error(yhat_xtrain, y_train) / 2
cv_MSE = mean_squared_error(yhat_xcv, y_cv) / 2

print("************** 2nd Order Polynomial **************")
print(f"MSE of training_set: {training_MSE: 0.2f}\nMSE of dev_set: {cv_MSE: 0.2f}")