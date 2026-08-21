import pandas as pd
import seaborn as sns

df = sns.load_dataset("penguins")

print(df.info())
print(df.isna().sum())
