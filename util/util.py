import enum
import numpy as np
import pandas as pd

def markOutliers(values):
    if (type(values)==list):
        values = np.array(values)
    sd = np.std(values)
    xbar = np.mean(values)
    datamin = xbar - ( 3 * sd )
    datamax = xbar + ( 3 * sd )
    return ((values < datamin) | (values > datamax))

def removeOutliers(df):
    for col in df.columns:
        df = df[markOutliers(df[col]) == False]
    return df
