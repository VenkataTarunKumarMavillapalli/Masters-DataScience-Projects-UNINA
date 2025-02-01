Statistical Data Analysis on Human Activity Recognition Data Set - April 8th 2023 

We have understood the data and features in the Human Activity Recognition Dataset, Defined a problem statement(The goal is to help in building an app for Healthcare Monitoring which uses sensors(accelorometer, gyroscope) to collect data from our mobile phone to predict Physical Activity accurately. We used the few stastical measures to find the final model which best suits and predicts activity correctly. Which can positively impact the lives of millions of people across different industries, including healthcare, sports, fitness. ). 

Performed Data Pre- Processing and Cleaning to exclude if there are any missing values, checked for outliers to remove bad data, used descriptive statistics to providing basic information about variables in a dataset and highlighting potential relationships between variables using graphical/pictorial Methods. We also tried Multidimensional Data Analysis where we used Hierarchical and K-means Clustering to find distinct group within our data and then done correlation analysis for dimensionality reduction has our dataset is High dimensional data.
Then we have used Lasso Regression for Feature selection. 

In Classification we used Support Vector Machine, Logistic Regression, Decision Tree and Random forest, where me constructed confusion matrix for all the models and used Accuracy, precision and Kappa statistics, recall and f-1 score has our statistical measures to find the best model which suites for the problem. Finally we performed K-fold cross validation to validate if the model is not overfitted.

Data Set Information: (https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) <- You can download dataset from here.

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

Check the README.txt file for further details about this dataset.

A video of the experiment including an example of the 6 recorded activities with one of the participants can be seen in the following link: [Web Link]

An updated version of this dataset can be found at [Web Link]. It includes labels of postural transitions between activities and also the full raw inertial signals instead of the ones pre-processed into windows.


Attribute Information:

For each record in the dataset it is provided:
- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope.
- A 561-feature vector with time and frequency domain variables.
- Its activity label.
- An identifier of the subject who carried out the experiment.

