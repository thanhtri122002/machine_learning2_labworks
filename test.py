
# One-Hot Encoding and Returning a DataFrame
import pandas as pd

df = pd.DataFrame({
    'Name': ['Joan', 'Matt', 'Jeff', 'Melissa', 'Devi'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
    'House Type': ['Apartment', 'Detached', 'Apartment', None, 'Semi-Detached']
    })
print(type(df))
ohe = pd.get_dummies(data=df, columns=['Gender'])
print(ohe)
print(type(ohe))