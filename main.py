import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Read the Auto data
csv_pandas = pd.read_csv('auto.csv')

# Output the first few rows
print(csv_pandas.head())
print()

# Output the dimensions of the data
print(csv_pandas.shape)
print()

# Data exploration with code
print(csv_pandas[['mpg', 'weight', 'year']].describe())
print()

# Explore data types
print(csv_pandas.dtypes)
print()

csv_pandas['cylinders'] = csv_pandas['cylinders'].astype('category')
csv_pandas['origin'] = pd.Categorical(csv_pandas['origin'])

print(csv_pandas.dtypes)
print()

# Deal with NAs
csv_pandas.dropna(inplace=True)
print(csv_pandas.shape)
print()

# Modify columns
csv_pandas['mpg_high'] = np.where(csv_pandas['mpg'] > csv_pandas['mpg'].mean(), 1, 0)
csv_pandas.drop(columns=['mpg', 'name'], inplace=True)
print(csv_pandas.head())
print()

# Data exploration with graphs
sns.catplot(x='mpg_high', kind='count', data=csv_pandas)
sns.relplot(x='horsepower', y='weight', hue='mpg_high', style='mpg_high', data=csv_pandas)
sns.boxplot(x='mpg_high', y='weight', data=csv_pandas)

# Train/test split
X = csv_pandas.drop(columns=['mpg_high'])
y = csv_pandas['mpg_high']
training_for_x, testing_for_x, training_for_y, testing_for_y = train_test_split(X, y, test_size=0.2, random_state=4321)

print(training_for_x.shape, testing_for_x.shape)

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(training_for_x, training_for_y)

predicted_y = log_reg.predict(testing_for_x)

print(classification_report(testing_for_y, predicted_y))

# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(training_for_x, training_for_y)

predicted_y = dtc.predict(testing_for_x)

print(classification_report(testing_for_y, predicted_y))

# Neural Network
neuralNetwork = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
neuralNetwork.fit(training_for_x, training_for_y)

predicted_y = neuralNetwork.predict(testing_for_x)

print(classification_report(testing_for_y, predicted_y, zero_division=0))

neuralNetwork2 = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500)
neuralNetwork2.fit(training_for_x, training_for_y)

predicted_y = neuralNetwork2.predict(testing_for_x)

print(classification_report(testing_for_y, predicted_y))

# Analysis