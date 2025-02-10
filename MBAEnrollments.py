#!/usr/bin/env python
# coding: utf-8

# In[392]:


import pandas as pd
import numpy as np
data = pd.read_csv('mba_decision_dataset.csv')
data.head()


# In[394]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, accuracy_score
import statsmodels.api as sm

# We need to convert categorical variables into a numerical format we can analyze. Let's start with dummies for binary categorical variables

# Target variable: create dummy variable for "Decided to Pursue MBA?"
def pursuedmba(pursued): 
    if isinstance(pursued, str) and pursued.strip().lower() == 'yes': 
        return 1 # Pursued an MBA 
    return 0 # Did not pursue an MBA

data['pursued'] = data['Decided to Pursue MBA?'].apply(pursuedmba)

# Create dummy variable for "Has Management Experience"
def hasmanagementexperience(hasmanagementexperience): 
    if isinstance(hasmanagementexperience, str) and hasmanagementexperience.strip().lower() == 'yes': 
        return 1 # Has management experience 
    return 0 # Has no management experience

data['mgmt'] = data['Has Management Experience'].apply(hasmanagementexperience)

# Function to create a dummy variable for gender
def gender(gender_value): 
    # Check if the value is 'Female' (case-insensitive)
    if isinstance(gender_value, str) and gender_value.strip().lower() == 'female': 
        return 1  # Female
    return 0  # Male

# Apply the function to the 'Gender' column to create a new 'female' column
data['female'] = data['Gender'].apply(gender)

# Create dummy variable for "Online vs. On-Campus MBA"
def online(online): 
    if isinstance(online, str) and online.strip().lower() == 'Online': 
        return 1 # Online
    return 0 # On-campus

data['online'] = data['Online vs. On-Campus MBA'].apply(online)

# Create dummy variable for "Location Preference (Post-MBA)"
def international(international): 
    if isinstance(international, str) and international.strip().lower() == 'International': 
        return 1 # International post-MBA
    return 0 # Domestic post-MBA

data['international'] = data['Location Preference (Post-MBA)'].apply(international)

# Check the first few rows
data.head()


# In[396]:


# Now let's create dummies for non-binary categorical variables using one-hot encoding

encoded_cols = pd.get_dummies(
    data[['Current Job Title', 'Undergraduate Major', 'Reason for MBA','MBA Funding Source','Desired Post-MBA Role']], 
    prefix=['current', 'undergrad', 'reason','funded','desired']
).astype(int)  # Convert encoded columns from boolean T/F to binary integers

# Join the new encoded variables with the data
data = pd.concat([data, encoded_cols], axis=1)


# In[398]:


# Let's try a variety of prediction methods. We'll begin with a simple OLS regression

# Convert monetary variables to more digestable units
data['Annual Salary (Before MBA)'] = data['Annual Salary (Before MBA)'] / 1000
data['Expected Post-MBA Salary'] = data['Expected Post-MBA Salary'] / 1000

# Check regressors for collinearity
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = data[['Age', 'Undergraduate GPA', 'Years of Work Experience', 'Annual Salary (Before MBA)', 'GRE/GMAT Score', 'Undergrad University Ranking', 'Entrepreneurial Interest', 'Networking Importance', 'Expected Post-MBA Salary', 'mgmt', 'female', 'online', 'international', 'current_Analyst', 'current_Consultant', 'current_Engineer', 'current_Entrepreneur', 'current_Manager', 'undergrad_Arts', 'undergrad_Business', 'undergrad_Economics', 'undergrad_Engineering', 'undergrad_Science', 'reason_Career Growth', 'reason_Entrepreneurship', 'reason_Networking', 'reason_Skill Enhancement','funded_Employer', 'funded_Loan', 'funded_Scholarship', 'funded_Self-funded', 'desired_Consultant', 'desired_Executive', 'desired_Finance Manager', 'desired_Marketing Director', 'desired_Startup Founder'
]]

# Add constant to the model (for intercept)
X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the result
print(vif_data)

target = 'pursued'

# Define the independent variables
independent_vars = [
    'Age', 'Undergraduate GPA', 'Years of Work Experience', 'Annual Salary (Before MBA)', 'GRE/GMAT Score', 'Undergrad University Ranking', 'Entrepreneurial Interest', 'Networking Importance', 'Expected Post-MBA Salary', 'mgmt', 'female', 'online', 'international', 'current_Analyst', 'current_Consultant', 'current_Engineer', 'current_Entrepreneur', 'current_Manager', 'undergrad_Arts', 'undergrad_Business', 'undergrad_Economics', 'undergrad_Engineering', 'undergrad_Science', 'reason_Career Growth', 'reason_Entrepreneurship', 'reason_Networking', 'reason_Skill Enhancement','funded_Employer', 'funded_Loan', 'funded_Scholarship', 'funded_Self-funded', 'desired_Consultant', 'desired_Executive', 'desired_Finance Manager', 'desired_Marketing Director', 'desired_Startup Founder'
]

# Ensure the independent variables exist in the dataset
data = data.dropna(subset=[target] + independent_vars)  # Drop rows with missing values

# Define the X (independent variables) and Y (target variable)
X = data[independent_vars]
y = data[target]

# We have a collinearity problem with the one-hot encoded dummy variables, so let's drop the constant to avoid having to drop any variables
# X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# Abysmal predictor! There seems to be perfect collinearity among encoded variables. Let's move on


# In[400]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = data[independent_vars]  # Independent variables
y = data[target]  # Dependent variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Lasso regression with cross-validation to choose the best alpha
lasso = LassoCV(cv=10)  # 10-fold cross-validation
lasso.fit(X_train, y_train)

# Get the best alpha
print(f"Best alpha: {lasso.alpha_}")

# Step 5: Predict using the test set
y_pred = lasso.predict(X_test)

# Step 6: Evaluate the model (using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 7: Get the coefficients of the model
coefficients = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# That output isn't very helpful for predictions. Let's move on to KNN


# In[402]:


# Let's try a KNN classifier

selected_columns = [
    'pursued','Age', 'Undergraduate GPA', 'Years of Work Experience', 'Annual Salary (Before MBA)', 'GRE/GMAT Score', 'Undergrad University Ranking', 'Entrepreneurial Interest', 'Networking Importance', 'Expected Post-MBA Salary', 'mgmt', 'female', 'online', 'international', 'current_Analyst', 'current_Consultant', 'current_Engineer', 'current_Entrepreneur', 'current_Manager', 'undergrad_Arts', 'undergrad_Business', 'undergrad_Economics', 'undergrad_Engineering', 'undergrad_Science', 'reason_Career Growth', 'reason_Entrepreneurship', 'reason_Networking', 'reason_Skill Enhancement','funded_Employer', 'funded_Loan', 'funded_Scholarship', 'funded_Self-funded', 'desired_Consultant', 'desired_Executive', 'desired_Finance Manager', 'desired_Marketing Director', 'desired_Startup Founder'
]
knndata = data[selected_columns]

# Separate features and target variable
X = knndata.drop(columns=['pursued'])
y = knndata['pursued']

# Check if dataset has enough samples
if len(knndata) < 1:
    raise ValueError("The dataset is empty after preprocessing")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize features using StandardScaler for kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Print the accuracy and classification report as a table
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(report_df)


# In[414]:


# Now let's try a Naive Bayes classifier on the same data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# First split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = nb.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Output results
print(f"Naïve Bayes Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report_df)


# In[416]:


from sklearn.model_selection import cross_val_score
# Cross validation

cv_scores = cross_val_score(knn, X, y, cv=5)  # 5-fold cross-validation

print("KNN Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

k_values = [1, 2,3, 4,5, 6,7, 8,9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"Accuracy with k={k}: {accuracy_score(y_test, y_pred)}")

# Cross-validation for Naïve Bayes
nb = GaussianNB()
nb_scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy') 
print("Naïve Bayes Mean Accuracy:", nb_scores.mean())

# The Naïve Bayes classifier provides the best predictor for whether someone will pursue an MBA, with an accuracy of about 57%


# In[418]:


# We can rank the features by their influence on whether the individual pursued an MBA

# Compute feature influence using mean differences
feature_means_0 = X_train[y_train == 0].mean(axis=0)
feature_means_1 = X_train[y_train == 1].mean(axis=0)
feature_influence = (feature_means_1 - feature_means_0).values  # Convert to NumPy array

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Influence': feature_influence})
feature_importance = feature_importance.sort_values(by='Influence', ascending=False)

# Isolate the ceteris paribus effect of being female
if 'female' in data:
    female_effect = feature_importance[feature_importance['Feature'] == 'female']
    print("Effect of Being Female:")
    print(female_effect)

# Gender disparity test using Chi-square
contingency_table = pd.crosstab(data['female'], data['pursued'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square test for gender disparity: chi2={chi2:.3f}, p-value={p:.3f}")

# Display evaluation metrics
print(f"Model Accuracy: {accuracy:.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display most influential features
print("Top Features Likely to Make Target = 1:")
print(feature_importance.head(82))

# This output's binary features suggest that advertising should target prospective candidates are funded by loans, value networking, were economics majors as undergrads, and currently work as consultants
# Our chi-squared test also suggests no statistically significant gender disparity. The feature importance ranking also shows that the female coefficient's influence is zero


# In[424]:


from scipy.stats import chi2_contingency

# That sounds questionable – gender has no effect? 

nb.fit(X_train_scaled, y_train)

# Group by gender (female) and calculate the sum and count for the pursued MBA column
gender_groups = data.groupby('female')['pursued'].agg(['sum', 'count'])

# Calculate the percentage of each gender who pursued an MBA
gender_groups['percentage'] = (gender_groups['sum'] / gender_groups['count']) * 100

# Map the gender (0 -> Male, 1 -> Female) for better readability
gender_groups.index = gender_groups.index.map({0: 'Male', 1: 'Female'})

# Display the result with the percentage
print(gender_groups[['percentage']])

# Let's use our Naive Bayes to compute the mean difference in predicted probabilities between men and women

# Ensure X_train and X_test have the same features
common_features = X_train.columns.intersection(X_test.columns)  # Find matching features
X_train_aligned = X_train[common_features]
X_test_aligned = X_test[common_features]

# Convert to NumPy array if model was trained on NumPy arrays
X_test_np = X_test_aligned.to_numpy()

# Ensure "female" column exists in X_test
if 'female' in X_test.columns:
    # Create two versions of X_test: one assuming male (female=0) and one assuming female (female=1)
    X_test_male = X_test.copy()
    X_test_female = X_test.copy()
    
    X_test_male['female'] = 0  # Simulating all as male
    X_test_female['female'] = 1  # Simulating all as female

    # Ensure both X_test_male and X_test_female only have the same columns as X_train
    X_test_male_aligned = X_test_male[common_features]
    X_test_female_aligned = X_test_female[common_features]

    # Convert to NumPy arrays for prediction
    X_test_male_np = X_test_male_aligned.to_numpy()
    X_test_female_np = X_test_female_aligned.to_numpy()

    # Get predicted probabilities from the Naive Bayes model
    probs_male = nb.predict_proba(X_test_male_np)
    probs_female = nb.predict_proba(X_test_female_np)

    # Compute the mean difference in predicted probabilities
    female_influence = np.mean(probs_female - probs_male, axis=0)

    # Create a DataFrame for easy interpretation
    female_effect = pd.DataFrame({'Class': nb.classes_, 'Influence': female_influence})
    
    print("Effect of Being Female on Predicted Probabilities:")
    print(female_effect)
else:
    print("⚠️ 'Female' column not found in X_test.")
    
# This corroborates the claim that gender has very little effect on whether a candidate will pursue an MBA in our predictive model.


# In[426]:


# Let's try it another way

# Create two identical candidate profiles, differing only in gender
example_male = X_train.iloc[0].copy()
example_female = X_train.iloc[0].copy()

example_male['female'] = 0
example_female['female'] = 1

# Scale the profiles using the same scaler
example_male_scaled = scaler.transform(example_male.values.reshape(1, -1))
example_female_scaled = scaler.transform(example_female.values.reshape(1, -1))

# Predict probabilities
prob_male = nb.predict_proba(example_male_scaled)[0][1]
prob_female = nb.predict_proba(example_female_scaled)[0][1]

print(f"Probability (male): {prob_male:.3f}")
print(f"Probability (female): {prob_female:.3f}")
print(f"Gender effect: {prob_female - prob_male:.3f}")

# Once again, we see that gender has very little effect on whether a candidate will pursue an MBA, and thus it shouldn't create a disparity in our advertising


# In[ ]:




