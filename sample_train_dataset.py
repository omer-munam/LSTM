import random
import pandas as pd
import numpy as np

df = pd.DataFrame(index=range(12000),columns=list(range(1,5)))
df = df.fillna("")

for index, rows in df.iterrows():
    df.at[index, 1] = random.randint(0,1) 
    df.at[index, 2] = random.randint(0,5)
    df.at[index, 3] = random.randint(0,1)
    df.at[index, 4] = random.randint(0,4)

export_csv = df.to_csv('./train/trainX.csv', header=False, index=False) 

df2 = pd.DataFrame(index=range(50))
df2 = df2.fillna("")

for index, rows in df2.iterrows():
    for j in range(1,6):
        df2.at[index, j] = str(random.randint(0,1))

# print(df2)

export_csv = df2.to_csv('./train/trainY.csv', header=False, index=False) 