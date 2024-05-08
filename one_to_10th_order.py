from my_common_imports import *
from loads_and_split_data import LoadAndSplitData

# Load and  split the  data
x_train, y_train, x_cv, y_cv, x_test, y_test = LoadAndSplitData()

# Initialize lists to save the errors, models, and feature transforms
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

for i in range(1, 11):
    # Add polynomial features
    poly = PolynomialFeatures(degree=i, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)
    x_cv_mapped = poly.transform(x_cv)
    polys.append(poly)

    # Feature scaling
    poly_scalar = StandardScaler()
    x_train_mapped_scaled = poly_scalar.fit_transform(x_train_mapped)
    x_cv_mapped_scaled = poly_scalar.transform(x_cv_mapped)
    scalers.append(poly_scalar)

    # Train the model
    model = LinearRegression()
    model.fit(x_train_mapped_scaled, y_train)
    models.append(model)

    # Compute MSE for training set & cross validation set
    yhat_xtrain = model.predict(x_train_mapped_scaled)
    yhat_xcv = model.predict(x_cv_mapped_scaled)

    training_MSE = mean_squared_error(yhat_xtrain, y_train) / 2
    cv_MSE = mean_squared_error(yhat_xcv, y_cv) / 2
    train_mses.append(training_MSE)
    cv_mses.append(cv_MSE)

    print(f"************** {i}th Order Polynomial **************")
    print(f"MSE of training_set: {training_MSE: 0.2f}\nMSE of dev_set: {cv_MSE: 0.2f}")
    
print('-------------------------------------------')
# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")
print('-------------------------------------------')

# Now for the test set

# Add polynomial features
x_test_mapped = polys[degree - 1].transform(x_test)

# Feature scaling
x_test_mapped_scaled = scalers[degree - 1].transform(x_test_mapped)

# MSE for test set
yhat_xtest = models[degree - 1].predict(x_test_mapped_scaled)
test_mse = mean_squared_error(yhat_xtest, y_test) / 2

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")