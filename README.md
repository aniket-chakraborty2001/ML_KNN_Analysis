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
I gone through some stages to finish the project and making it a well organized one. The stages of the project are well described and to the point. In the project, all the steps are done with notations, so that there will be no problem to understand each part of the project. The steps of the projevt are -

**1. Basic Data Exploration:**
In this step, I read the data set using pandas read_csv() function as df. Then I find the first 5 rows/observations of the data frame using .head() method. I also found the shape of the data frame (row and column number). I found the coulmn names and data types of each column. These are the basic data exploration step.

**2. Train and Test set defining:**
Now, I first define the feature set denoted by **x** and convert it in an array. Similarly, I define the test set, denoted by **y** and convert in an array. Then using **train_test_split of model_selection module of sklearn** , I break the data set into four parts. They are - two parts for train data set (train_x, train_y) and two for test data set (test_x and test_y). As discussed earlier, normalization is very important in KNN to eliminate any effect of different units used in the model. So we define train_x_norm and test_x_norm array with the **preprocessing module of sklearn package**. Now, these train_x_norm and test_x_norm became the training and testing set for independent variables to build the required model.

**3. Builing the First Model using K = 4 and Prediction and Model Accuracy Evaluation:**
Now, I build the first model using 4 nearest neighbourhoods using the **KNeighborsClassifier object of sklearn.neighbors module**. I fit the model with the training data(train_x_norm and train_y). After building the model, I use the .predict() method to get the predicted classes of the unseen test data using test_x_norm data set. The predicted class I get is 3,1,3,2,4. After this, I calculate the Accuracy of the Training data by fitting the train_x_norm data in the model and accuracy of testing data using the **accuracy_score() function of metrics module of sklearn package**. 

**4. Builing the First Model using K = 6 and Prediction and Model Accuracy Evaluation:**
Now, I build the first model using 6 nearest neighbourhoods using the **KNeighborsClassifier object of sklearn.neighbors module**. I fit the model with the training data(train_x_norm and train_y). After building the model, I use the .predict() method to get the predicted classes of the unseen test data using test_x_norm data set. The predicted class I get is 3,3,3,2,4. After this, I calculate the Accuracy of the Training data by fitting the train_x_norm data in the model and accuracy of testing data using the **accuracy_score() function of metrics module of sklearn package**. 

### Interpretation of Model Accuracy
I get the Model Accuracy on test data as 0.33(for k = 4) and 0.335(for k = 6) respectively. It is clear that Accuracy of model on test data is better for k = 6. It concluded that we build a grate model when k = 6

### Question to be in Mind
Then a question arises, how we can select the k value for which the test data accuracy will be maximum and what will be the accuracy value. To answer this, we can use two mwthods. The first method is by writing codes and comparing values. The second method is by graphical representation. 

#### **Method - 1: By Writing Codes:**
