# Project Name: ML_KNN_Analysis
Consider the Telecommunication data set. Available in GitHub and Kaggle. Using this we perform a supervised Machine Learning technique called Classification using KNN algorithm.

## Classification
A supervised Machine Learning algorithm, used to categorize some unknown items into a discete set of categories or classes. It uses the characteristics of the unknown item. For example, suppose a data set that consider whether a tumor cell in human body is bening or malignent. After observing say 100 data points, we train a model and using that we are prediting the next tumor will be bening or malignent depending on the features the tumor will have.

# KNN or K-Nearest Neighbours
A classification algorithm, used to predict the class of an unknowm object by observing the class of the neighbouring objects. So, looking only the 1 st neighbour is not best. So, in general we consider 6 or 7 nearest neighbours and observe the item that has most occurance. The unknown object will have the same class. KNN completly works on distance between two data points. This distance is not always Euclidian distance, it may be multi dimensional. For this reason, normalization in KNN is very important.

## About the Tele-Communication Data set
The data set that is considered here is already attached in the file section. The data set contains 1000 observations and 12 features. The last feature which is **custcat** contains the classification class. It contains 4 values namely 1, 2, 3 and 4. 1 indicates Basic Service, 2 indiactes E service, 3 and 4 indicates Plus service and Total service respectively. Other than the column **custcat** there are 11 columns. Each are considered as feature set.

## Basic Informations of Project and Libraries used
  1. I use different modules of Scikit-learn library.
  2. Scikit-learn is written as sklearn in python.
  3. sklearn package only works on arrays. So all data frames are needed to be converted to arrays.
  4. Normalization on independent part of train and test set is important to measure similarity.
  5. In train test split, general rule is maintained. 80% data for training and 20% data for testing.

## Project Stages
I gone through some stages to finish the project and making it a well organized one. The stages of the project are - 
##### 1. Basic Data Exploration:
In this step, I read the data set using pandas read_csv() function as df. 
