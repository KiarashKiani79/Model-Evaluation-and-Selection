from my_common_imports import *
from loads_and_split_data import LoadAndSplitData
from build_nn_models import BuildModels
import tensorflow as tf

#* Prepare the Data
# Load and  split the  data
X_train, y_train, X_cv, y_cv, X_test, y_test = LoadAndSplitData()

# Add Polynomial features
degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
X_train_mapped = poly.fit_transform(X_train)
X_cv_mapped = poly.transform(X_cv)
X_test_mapped = poly.transform(X_test)

# Feature scaling
poly_scalar = StandardScaler()
X_train_mapped_scaled = poly_scalar.fit_transform(X_train_mapped)
X_cv_mapped_scaled = poly_scalar.transform(X_cv_mapped)
X_test_mapped_scaled = poly_scalar.transform(X_test_mapped)


# Initialize the MSEs Containers
nn_train_mses = []
nn_cv_mses = []

#* Build & Train the Models
# Define the model
nn_models = BuildModels()

# Loop through each model
for model in nn_models:
    
    # Setup the optimizer=adam and loss=mse
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mse',
        )
    
    print(f"+-+-+-+-+\nTraining {model.name}...")
    
    # Train the model
    model.fit(X_train_mapped_scaled, y_train, epochs=300, verbose=0)
    
    print("Done!\n")
    
    # Record the training MSEs
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    nn_train_mses.append(train_mse)
    
    # Record the cross validation MSEs 
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    nn_cv_mses.append(cv_mse)
    
    
print("************** Neural Network Models **************")
for model_index in range (len(nn_models)):
    print(f"Model-{model_index + 1}: "
          + f"Training MSE: {nn_train_mses[model_index]: 0.2f}/"
          + f" /Cross Validation MSE: {nn_cv_mses[model_index]: 0.2f}\n"
          )

#* Select the Best Model
best_model_index = np.argmin(nn_cv_mses)
best_model = nn_models[best_model_index]

# Compute the test MSE
yhat = best_model.predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected Model: {best_model_index+1}")
print(f"Training MSE: {nn_train_mses[best_model_index]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[best_model_index]:.2f}")
print(f"Test MSE: {test_mse:.2f}")