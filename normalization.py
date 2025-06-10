#normalization techniques
from google.colab import files
import pandas as pd
import numpy as np
uploaded=files.upload()
df=pd.read_csv('Housing.csv')
features = df.drop('price', axis=1)
features = features.select_dtypes(include=[np.number])
features = features.fillna(features.mean())
def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


min_max_normalized_df = features.apply(min_max_normalize)
print("\nMin-Max Normalized Data:")
print(min_max_normalized_df)
def z_score_normalize(data):
    mean = data.mean()
    std_dev = data.std()
    return (data - mean) / std_dev

standardized_df = features.apply(z_score_normalize)
print("\nStandardized (Z-score) Data:")
print(standardized_df)
def max_abs_scale(data):
    max_abs_val = data.abs().max()
    return data / max_abs_val

max_abs_scaled_df = features.apply(max_abs_scale)
print("\nMax Abs Scaled Data:")
print(max_abs_scaled_df)
def robust_scale(data):
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return (data - median) / iqr
print("\nRobust Scaled Data:")
robust_scaled_df = features.apply(robust_scale)
print(robust_scaled_df)
