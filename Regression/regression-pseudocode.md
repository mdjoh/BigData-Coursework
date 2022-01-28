REGRESSION PSEUDOCODE

Load diabetes data using load_diabetes function within sklearn.datasets

Split the data into training and testing sets using train_test_split(data, target) in sklearn.model_selection
Return is data split into training data and test data and target splint into training target and test target

### LINEAR REGRESSION ###
Instantiate the model via model = LinearRegression() function in sklearn.linear_model
Fit the model via model.fit(x_train, target_train) function (ie. "training" the model in this step)
Test the trained model and generate test results using test data (ie. predict target from the test data using the trained model) using model.predict(X_test)
Score the model to see how good the regression the model is using score(model, predicted_target, expected_target) (score ranges from 0 to 1 with 0 being the worst and 1 being perfect)

Optional: scatter plot predicted vs. expected to visualize how predicted compares to expected values

### GRADIENT BOOSTING TREE REGRESSION ###
Repeat Linear Regression steps from above but instantiate the model via model = GradientBoostingRegressor() which is found in sklearn.ensemble
