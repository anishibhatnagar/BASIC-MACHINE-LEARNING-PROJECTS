#pre processing
from google.colab import files
import pandas as pd
uploaded=files.upload()
df=pd.read_csv('Buy_Computer.csv',encoding='latin-1')
#missing values
missing_values=df.isnull().sum()
print(missing_values)
#duplicates
duplicates=df[df.duplicated()]
print(duplicates)
#outliers
numerical_df = df.select_dtypes(include=['number'])
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((numerical_df < lower_bound) | (numerical_df > upper_bound)).sum()
print("\nNumber of outliers in each column:")
print(outliers)
