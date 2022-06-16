import numpy as np
import pandas as pd
import json
import geopy
from geopy import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle

amesRealEstate = pd.read_csv('../data/Ames_Real_Estate_Data.csv')

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

def returnDFWithISUDistance(df):
    with open('../data/googleApiKey.json') as d:
        googleKeyDictionary = json.load(d)
        apiKey = googleKeyDictionary['apikey']
    locator = GoogleV3(api_key=apiKey)
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1.5)
    ISU = geocode('Iowa State University, Ames, USA').point[0:2]
    pids = df['PID'].to_frame()
    pidMapRefNoMerge = pids.merge(amesRealEstate[['MapRefNo','Prop_Addr']],
           how='left',left_on='PID',right_on='MapRefNo')
    locations = [geocode(addr+', Ames,IA, USA').point[0:2] if type(addr)==str else None for addr in pidMapRefNoMerge['Prop_Addr']]
    pidMapRefNoMerge['ISUDistance'] = pd.Series([great_circle(house,ISU).mi for house in locations])
    df = df.merge(pidMapRefNoMerge[['MapRefNo','ISUDistance']],
           how='left',left_on='PID',right_on='MapRefNo')
    return df
