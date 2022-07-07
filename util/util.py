import numpy as np
import pandas as pd
import json
import geopy
from geopy import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from os.path import exists
from IPython.display import display, HTML
import sklearn.linear_model as lm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

amesRealEstate = pd.read_csv('../data/Ames_Real_Estate_Data.csv')
with open('../data/googleApiKey.json') as d:
    googleKeyDictionary = json.load(d)
    apiKey = googleKeyDictionary['apikey']
locator = GoogleV3(api_key=apiKey)
geocode = RateLimiter(locator.geocode, min_delay_seconds=1.5)
ISU = geocode('Iowa State University, Ames, USA').point[0:2]

def returnDFWithISUDistance(df,displayNoneNumber = False):
    
    pids = df['PID'].to_frame()
    pidMapRefNoMerge = pids.merge(amesRealEstate[['MapRefNo','Prop_Addr']],
           how='left',left_on='PID',right_on='MapRefNo')
    locations = [getLocation(addr) if type(addr)==str else None for addr in pidMapRefNoMerge['Prop_Addr']]
    if (displayNoneNumber):
        total = df.shape[0]
        noneCount = sum(x is None for x in locations)
        print(f'{100 * noneCount/total}% of data is None')
    pidMapRefNoMerge['ISUDistance'] = pd.Series([great_circle(house,ISU).mi if house != None else np.nan for house in locations])
    df = df.merge(pidMapRefNoMerge[['MapRefNo','ISUDistance']],
           how='left',left_on='PID',right_on='MapRefNo')
    return df

def getLocation(address):
    try:
        location = geocode(address+', Ames,IA, USA').point[0:2]
        return location
    except BaseException:
        return None

def replaceNansWithTrainingDataValues(df):
    with open('../data/trainNanReplacementValuesDictionary.json') as d:
        trainNanReplacementValuesDictionary = json.load(d)
    for col in df.columns:
        df[col] = df[col].fillna(trainNanReplacementValuesDictionary[str(col)])
    return df

def removeDummiesAndCorrelatedFeaturesFromAvailabilityList(availabilityList,feature):
    with open('../data/sigCorrDictionary.json') as d:
        sigCorrDictionary = json.load(d)
    with open('../data/relatedDummiesDictionary.json') as d:
        relatedDummiesDictionary = json.load(d)
    if (feature in relatedDummiesDictionary.keys()):
        for dummy in relatedDummiesDictionary[feature]:
            if dummy in availabilityList:
                availabilityList.remove(dummy)
    for corrFeature in sigCorrDictionary[feature]:
        if corrFeature in availabilityList:
            availabilityList.remove(corrFeature)
    if feature in availabilityList:
        availabilityList.remove(feature)
    return availabilityList

def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>") ) )

def IsHomoskedastic(X,y,name,boxCoxLambda = None):
    X = np.array(X).reshape(-1,1)
    y = np.array(y).reshape(-1, 1)
    linmodel = lm.LinearRegression()
    linmodel.fit(X,y)
    yPredict = [prediction[0] for prediction in linmodel.predict(X)]
    residualTable = pd.DataFrame({'yPredict': yPredict},
    columns = ['yPredict'])
    residualTable['residual'] = linmodel.predict(X) - y

    qValue = 3
    out, bins = pd.qcut(residualTable['yPredict'],q=qValue, duplicates='drop',retbins = True)
    residualTable['bin'] = pd.qcut(residualTable['yPredict'],q=qValue,
    labels=list(np.linspace(0,len(bins)-2,len(bins)-1)),duplicates='drop')

    isCentered = IsCentered(residualTable)
    spreadValue = retrieveSpreadValue(residualTable)
    hasUniformSTD = spreadValue < 1.5

    if(isCentered == False):
        drawRegression(X,y,linmodel.intercept_,linmodel.coef_[0],spreadValue, name, "notCentered", boxCoxLambda)
    if(hasUniformSTD == False):
        drawRegression(X,y,linmodel.intercept_,linmodel.coef_[0],spreadValue, name, "isHeteroskedastic", boxCoxLambda)
    if (isCentered and hasUniformSTD):
        drawRegression(X,y,linmodel.intercept_,linmodel.coef_[0],spreadValue, name, "isHomoskedastic", boxCoxLambda)
        checkNormality(residualTable['residual'],name,boxCoxLambda)

    return isCentered and hasUniformSTD

def drawRegression(X,y,b,m,spreadValue,name,folder,boxCoxLambda):
    plt.figure()
    plt.scatter(X,y)
    x = np.linspace(np.min(X),np.max(X),100)
    plt.plot(x,(m*x+b))
    plt.xlabel(name)
    if (boxCoxLambda == None):
        plt.ylabel("sales price")
        title = f'Sales Price vs {name}'
        plt.title(f'{title} spreadValue={spreadValue}')
        plt.savefig(f'../images/{folder}/{name}.jpg')
    else:
        plt.ylabel(f'boxcox(SalesPrice,{boxCoxLambda})')
        title = f'boxcox(SalesPrice,{boxCoxLambda}) vs {name}'
        plt.title(f'{title} spreadValue={spreadValue}')
        plt.savefig(f'../images/{folder}/{title}.jpg')
    plt.close()

def engineerFeature(xTrain,newX,y,name):
    newX = np.array(newX).reshape(-1,1)
    y = np.array(y).reshape(-1, 1)
    linmodel = lm.LinearRegression()
    linmodel.fit(newX,y)
    baseScore = linmodel.score(newX,y)

    notZero = newX != 0.0
    linmodel.fit(np.log(newX[notZero]).reshape(-1,1),np.log(y[notZero]).reshape(-1,1))
    power = np.round(linmodel.coef_[0],2)[0]
    transformedName = f'{name}^{power}'
    xPow = newX ** power
    if (np.isinf(xPow).any() or np.isnan(xPow).any()):
        powerScore = -np.inf
    else:
        linmodel.fit(xPow,y)
        powerScore = linmodel.score(xPow,y)

    if ((powerScore > baseScore) and IsHomoskedastic(xPow,y,transformedName)):
        xTrain[transformedName] = xPow
        return xTrain
    else:
        return TryBoxCox(xTrain,newX,y,name)

def engineerSmallFeature(xTrain,newX,y,name):
    newX = np.array(newX).reshape(-1,1)
    y = np.array(y).reshape(-1, 1)
    linmodel = lm.LinearRegression()
    linmodel.fit(newX,y)
    baseScore = linmodel.score(newX,y)

    notZero = newX != 0.0
    linmodel.fit(np.log(newX[notZero]).reshape(-1,1),np.log(y[notZero]).reshape(-1,1))
    power = np.round(linmodel.coef_[0],2)[0]
    powerName = f'{name}^{power}'
    xPow = newX ** power
    if (np.isinf(xPow).any() or np.isnan(xPow).any()):
        powerScore = -np.inf
    else:
        linmodel.fit(xPow,y)
        powerScore = linmodel.score(xPow,y)

    boxcox_y, best_lambda = boxcox(y.ravel())
    roundedLambda = round(best_lambda,2)
    boxcox_y = boxcox(y.ravel(),lmbda = roundedLambda)
    linmodel.fit(newX,boxcox_y.reshape(-1,1))
    m = np.round(linmodel.coef_[0],2)[0]
    b = np.round(linmodel.intercept_,2)[0]
    boxCoxScore = linmodel.score(newX,boxcox_y.reshape(-1,1))

    bestScore = np.max([baseScore,powerScore,boxCoxScore])
    if bestScore == powerScore:
        xTrain[powerName] = xPow
        IsHomoskedastic(xPow,y,powerName)
    elif bestScore == boxCoxScore:
        transformed_new_x = InvBoxCox(newX,roundedLambda,m,b)
        transformedName = f'{name}_invbc_l{roundedLambda}_m{m}_b{b}'
        xTrain[transformedName] = transformed_new_x
        IsHomoskedastic(transformed_new_x,y,transformedName)
    else:
        xTrain[name] = newX
        IsHomoskedastic(newX,y,name)
    return xTrain


def TryBoxCox(xTrain,newX,y,name):
    linmodel = lm.LinearRegression()
    linmodel.fit(newX,y)
    baseScore = linmodel.score(newX,y)
    
    transformed_y, best_lambda = boxcox(y.ravel())
    roundedLambda = round(best_lambda,2)
    transformed_y = boxcox(y.ravel(),lmbda = roundedLambda)
    linmodel.fit(newX,transformed_y.reshape(-1,1))
    boxCoxScore = linmodel.score(newX,transformed_y.reshape(-1,1))

    if ((boxCoxScore > baseScore) and IsHomoskedastic(newX,transformed_y,name,roundedLambda)):
        m = np.round(linmodel.coef_[0],2)[0]
        b = np.round(linmodel.intercept_,2)[0]
        transformed_new_x = InvBoxCox(newX,roundedLambda,m,b)
        transformedName = f'{name}_invbc_l{roundedLambda}_m{m}_b{b}'
        if IsHomoskedastic(transformed_new_x,y,transformedName):
            xTrain[transformedName] = transformed_new_x
            return xTrain 
        else:
            return None
    else:
        return None

def InvBoxCox(X,l,m,b):
    if (l == 0.0):
        return np.exp(m * X + b)
    else:
        return (l*(m * X + b) + 1) ** (1/l)

def IsCentered(residualTable):
    bins = list(set(residualTable['bin']))
    for bin in bins:
        binnedValues = residualTable[residualTable['bin'] == bin]
        low = binnedValues['residual'].quantile(0.025)
        high = binnedValues['residual'].quantile(0.975)
        if (low > 0.0 or high < 0.0):
            return False
    return True

def retrieveSpreadValue(residualTable):
    varianceValues = []
    bins = list(set(residualTable['bin']))
    for bin in bins:
        binnedValues = residualTable[residualTable['bin'] == bin]
        sd = np.std(binnedValues['residual'])
        varianceValues.append(sd ** 2)
    return (np.max(varianceValues)/np.min(varianceValues))

def checkNormality(values,name,boxCoxValue):
    fig,axs = plt.subplots(2,1)
    sns.histplot(values,kde=True,ax=axs[0])
    qqplot(values,line='s',ax=axs[1])
    if (boxCoxValue == None):
        imageTitle = f'{name} Normal Errors Check'
        plt.savefig(f'../images/normalCheck/{imageTitle.replace(" ","")}.jpg')
    else:
        imageTitle = f'boxcox({boxCoxValue}) vs {name} Normal Errors Check'
        plt.savefig(f'../images/normalCheck/{imageTitle.replace(" ","")}.jpg')
    plt.close()

def hasInfOrNanValues(arr):
    np.isnan(arr).any()

def plotCorrelation(x,y,name):
    plt.figure()
    plt.scatter(x,y)

    corrVal,pVal = stats.spearmanr(x,y)

    linmodel = lm.LinearRegression()
    linmodel.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
    m = linmodel.coef_[0]
    b = linmodel.intercept_
    fitX = np.linspace(min(x),max(x),100)
    plt.plot(fitX,m*fitX+b)
    plt.title(f'{name} R={round(corrVal,2)}')
    plt.savefig(f'../images/sigCorrs/{name}.jpg')
    plt.close()
    