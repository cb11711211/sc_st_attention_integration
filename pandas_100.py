# %%
import numpy as np
import pandas as pd

data = {
    'size': ['XL','L','M', np.nan, 'M', 'M'],
    'color': ['red', 'blue', 'green', 'red', 'green', 'blue'],
    'gender': ['F','M','F',np.nan,'F','M'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 400, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame.from_dict(data)
df.isna().sum() / len(df)
# %%
df[['weight']].fillna(df[['weight']].mean())
# %% Use SimpleImputer to impute NA value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['weight']] = imputer.fit_transform(df[['weight']])
print(f'The imputer value is: ',imputer.statistics_)
# %%
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=99)
df[['price']] = imputer.fit_transform(df[['price']])
# %%
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df[['size']] = imputer.fit_transform(df[['size']])
# %%
