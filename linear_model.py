from my_common_imports import *
from loads_and_split_data import LoadAndSplitData

# Load & split the data 
x_train, y_train, x_cv, y_cv, x_test, y_test = LoadAndSplitData()

# Feature scaling
scalar_linear = StandardScaler()
x_train_scaled = scalar_linear.fit_transform(x_train)

x_cv_scaled = scalar_linear.transform(x_cv)

# Train the model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Evaluate the model
yhat_train = model.predict(x_train_scaled)
yhat_cv = model.predict(x_cv_scaled)

# Calculte Mean Square Error
MSE_train = mean_squared_error(y_train, yhat_train) / 2
MSE_cv = mean_squared_error(y_cv, yhat_cv) / 2

print("************** linear Model **************")
print(f"MSE of training_set: {MSE_train: 0.2f}\nMSE of dev_set: {MSE_cv: 0.2f}")
