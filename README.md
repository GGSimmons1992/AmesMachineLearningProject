# AmesMachineLearningProject

House prices vary on various different features attached to each house. This project focuses on the training of a linear regression and a random forest regressor. 

## data (not included. Using .gitignore)

Due to data size and standard practice, my dataset is not included in this repo. I am using Ames Housing / Real Estate data, which is easily procurable. This folder also contains csv files of train-test-split data with and witout dummy varriables appended. In addition to that there are jsons used to train the linear regression model in machineLearning.ipynb and store the googleApiKey.

## images

Images contains 4 subfolders -- isHeteroskedastic, isHomoSkedastic, normalCheck, and notCentered -- and 2 images, MSEScores.png and TreeFeatureImportance.png. Scatter plots and attempted regression lines of continuous varriable data with spreadValues above 1.5 are stored in isHeteroskedastic; spreadValues are calulated by retrieveSpreadValue in util/util.py. Scatter plots and attempted regression lines of continuous varriable data that fail centering test defined by IsCentered in util/util.py are stored in notCentered. Continuous varriable data with spreadValues under 1.5 and pass the IsCentered test have their scatter plots and regression lines stored in isHomoSkedastic and have their histograms and qqplots stored in normalCheck. TreeFeatureImportances.png shows the top 10 features and their featureImportances determined by the RandomForestRegressor use in machineLearning.ipynb. MSEScores.png displays the mean square error of training and testing sets produced by the linear regression and the random forest regressor in machineLearning.png

## notebooks

eda.ipynb is my initial eda of the data and a ground for testing python funcationality

machineLearning.ipynb is my pipeline used to add dummies to the original dataset, train-test-split the dataset, and train and test the linear regression model and the RandomForestRegressor. 

## util

util.py is a custom python module that notebooks use.

## License

This work uses a MIT License, granting people to use or reuse this project for their own purposes.