import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

df = pd.read_csv('perth.csv')
print(df.isnull().sum())  # check if any empty values

# Boxplot of price
sns.boxplot(data=df[['PRICE']])
plt.show()

# Histogram of price
sns.histplot(df['PRICE'], kde=True)
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of bedrooms vs price
sns.scatterplot(x=df['BEDROOMS'], y=df['PRICE'], data=df)
plt.show()

# Separate into numerical snd categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64'], exclude=['bool']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns)

# Correlation matrix
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Simple Linear Regression
x = df['BEDROOMS']  # Predictor
y = df['PRICE']  # Response
x = sm.add_constant(x)  # Add a constant to the predictor variable (for the intercept)
model = sm.OLS(y, x).fit()  # Fit the model
plt.scatter(df['BEDROOMS'], df['PRICE'], color='blue', label='Data')
plt.plot(df['BEDROOMS'], model.predict(x), color='red', label='Regression Line')
plt.xlabel('BEDROOMS')
plt.ylabel('PRICE')
plt.title('Simple Linear Regression: Bedrooms vs. House Price')
plt.legend()
plt.show()

# Confidence interval
mean = np.mean(df['PRICE'])
std_dev = np.std(df['PRICE'], ddof=1)
confidence_level = 0.95
n = len(df)
t_critical = st.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
margin_of_error = t_critical * std_dev / np.sqrt(n)  # Calculates the margin of error for the confidence interval
ci = (mean - margin_of_error, mean + margin_of_error)
print(ci)

# Standardize and visualize bedrooms
df_std = pd.read_csv('perth.csv')
num_col = df_std.select_dtypes(include=['int64', 'float64'], exclude=['bool']).columns
df_std[numerical_columns] = StandardScaler().fit_transform(df_std[num_col])
sns.boxplot(data=df_std['BEDROOMS'])
plt.show()
