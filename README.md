# AmesMachineLearningProject

Insert summary here

## data (not included. Using .gitignore)

Due to data size and standard practice, my dataset is not included in this repo. I am using Ames Housing / Real Estate data, which is easily procurable. This folder also contains csv files of train-test-split data with and witout dummy varriables appended. In addition to that there are jsons used to train the linear regression model in machineLearning.ipynb and store the googleApiKey.

## images

Images contains 4 subfolders -- isHeteroskedastic, isHomoSkedastic, normalCheck, and notCentered -- and 2 images, MSEScores.png and TreeFeatureImportance.png. Scatter plots and attempted regression lines of continuous varriable data with spreadValues above 1.5 are stored in isHeteroskedastic; spreadValues are calulated by retrieveSpreadValue in util/util.py. Scatter plots and attempted regression lines of continuous varriable data that fail centering test defined by IsCentered in util/util.py are stored in notCentered. Continuous varriable data with spreadValues under 1.5 and pass the IsCentered test have their scatter plots and regression lines stored in isHomoSkedastic and have their histograms and qqplots stored in normalCheck. TreeFeatureImportances.png shows the top 10 features and their featureImportances determined by the RandomForestRegressor use in machineLearning.ipynb. MSEScores.png displays the mean square error of training and testing sets produced by the linear regression and the random forest regressor in machineLearning.png

## notebooks

dataProcessor.ipynb is my pipeline used to generate all figures. First, ensure all data mentioned in the data section above is in a data folder (which should be at the same level of the other folders). Second, make sure the names and paths of .csv file in the Data Manipulation section match up with the .csv files in the data folder (See last sentence in data section of this readme). Once the data is in the data folder and the paths in the read_csv commands are verified to match the data in the data folder, one can run the whole notebook, to populate the images folders with the correct images.

## src

utils.py is a custom python module that notebooks/dataProcess.ipynb uses.

## License

This work uses a MIT License, granting people to use or reuse this project for their own purposes.