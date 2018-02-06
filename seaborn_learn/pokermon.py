# -*- coding: utf-8 -*-
# Pandas for managing datasets
import pandas as pd
# Matplotlib for additional customization
from matplotlib import pyplot as plt
# %matplotlib inline
# Seaborn for plotting and styling
import seaborn as sns

# Read dataset
df = pd.read_csv('Pokemon.csv', index_col=0)
# Display first 5 observations
df.head()

# Recommended way
sns.lmplot(x='Attack', y='Defense', data=df)
 
# Alternative way
# sns.lmplot(x=df.Attack, y=df.Defense)