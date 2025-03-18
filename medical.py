import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

medicalData = pd.read_csv('insurance.csv')

# Data analysis 
# print(medicalData.describe())
# print(medicalData.isnull().sum())

# Data Visualisation 
# 1) Age Distribution 
plt.figure(figsize=(4,4))
sns.displot(medicalData['age'])
plt.title("Age Distribution")


# 2) Gender Distribution
plt.figure(figsize=(4,4))
sns.countplot(x = 'sex', data= medicalData)
plt.title("Gender Distribution")


# 3) Bmi distribution
plt.figure(figsize=(4,4))
sns.displot(medicalData['bmi'])
plt.title("BMI Distribution")
# plt.show()


# Feature Encoding
# 1) Male --> 0  Female --> 1
# 2) Smoker, Yes--> 0, No -->1
medicalData.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medicalData.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medicalData.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
# print(medicalData.head())

# Separating features and price
x = medicalData.drop(columns='charges')
y = medicalData['charges']

# Splittng data into test and training 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Model 1) Linear Regression
model1 = LinearRegression()
model1.fit(x_train, y_train)

# R2 Score (Training) (Linear Regression)
trainPredict = model1.predict(x_train)
r2ScoreTrain = metrics.r2_score(y_train, trainPredict)
print("Linear Regression Training R2 Score", r2ScoreTrain)

# R2 Score (Testing) (Linear Regression)
testPredict = model1.predict(x_test)
r2ScoreTest = metrics.r2_score( y_test, testPredict)
print("Linear Regression Test Dataset R2 Score", r2ScoreTest)

# Model 2) XGBRegressor
model2 = XGBRegressor()
model2.fit(x_train, y_train)

# R2 Score (Training) (XGBRegressor)
trainPredict = model2.predict(x_train)
r2ScoreTrain = metrics.r2_score(y_train, trainPredict)
print("XGBRegressor Training R2 Score", r2ScoreTrain)

# R2 Score (Testing)
testPredict = model2.predict(x_test)
r2ScoreTest = metrics.r2_score(y_test, testPredict)
print("XGBRegreesor Test Dataset R2 Score", r2ScoreTest)

# Predictive model
# Input from user
print("Enter the following details to predict insurance charges:")

age = int(input("Age: "))
sex = input("Sex (male/female): ")
bmi = float(input("BMI: "))
children = int(input("Number of children: "))
smoker = input("Smoker (yes/no): ")
region = input("Region (southeast/southwest/northeast/northwest): ")

# Encoding input values
sex = 0 if sex == 'male' else 1
smoker = 0 if smoker == 'yes' else 1
region_dict = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
region = region_dict.get(region, -1)  # returns -1 if region is invalid

# Check for valid region
if region == -1:
    print("Invalid region entered.")
else:
    # Create input array
    input_data = np.array([[age, sex, bmi, children, smoker, region]])

    # Make prediction using XGBoost model
    prediction = model2.predict(input_data)
    print(f"Predicted Insurance Charges: ₹{prediction[0]:.2f}")
