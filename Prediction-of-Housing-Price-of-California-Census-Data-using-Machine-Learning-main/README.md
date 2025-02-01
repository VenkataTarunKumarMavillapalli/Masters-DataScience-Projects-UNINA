feb 2023

---------------

Predicting Housing Price of California Census Data using Machine Learning

Our task is to use the California census data to build a model to find the housing prices in the state. This data includes features such as Population, Median Income and Median housing price for each block group in California.

A block group has a population between 600 to 3,000. We will call them "districts" for short. Our model should be able to predict the median housing price for any district.

We have performed Exploration of California census data, Data Pre-Processing, Data Preparation, Model Development and Training, Model Evaluation and Validation, Fine Tuning using Hyper Parameters

We have measured the performance of different models using the Root Mean Squared Error (RMSE). Also predicted the median_housing_price.

Random forest model was best suited for our data set. Hence used it for the fine tuning using grid search.

------------

Our task is to use the California census data to build a model to find the housing prices in the state. This data includes features such as Population, Median Income and Median housing price for each block group in California.

A block group has a population between 600 to 3,000. We will call them "districts" for short. Our model should be able to predict the median housing price for any district.

Also to measure the performance of the model we have used the Root Mean Squared Error (RMSE).

-------------
About the dataset

1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
10. oceanProximity: Location of the house w.r.t ocean/sea
