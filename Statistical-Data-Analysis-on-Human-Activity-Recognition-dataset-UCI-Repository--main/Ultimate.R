setwd("/Users/tarunkumar/Desktop")
library(randomForest)   # random forest
library(caret)          # machine learning functions
library(kernlab)        #provides a framework for creating and using kernel- based algorithms.
library(naivebayes)   #naivebayes
library(glmnet)       #cross-validation
library(ggplot2)      # visualization
library(dplyr)        # data wrangling
library(Rtsne)        # EDA3
library(e1071)          # naive bayes
library(rpart)          # decision tree
library(rattle)         # tree visualization
library(class)          # k-NN
library(forecast)       #displaying and analyzing univariate time series forecasts
library(tidyverse)     # Transform and better visualization of data 
library(corrplot)      # Plot for correlation 
library(Hmisc)         # High level graphics
library(rpart)
library(MLmetrics)      # machine learning metrics
library(mlbench)
library(Boruta)
library(dendextend)
library(plyr)
library(reshape2)
library(lattice)
library(MASS)
library(psych)




train_X<-read.table("UCI HAR Dataset/train/X_train.txt")

train_Y<-read.table("UCI HAR Dataset/train/y_train.txt")

test_X<-read.table("UCI HAR Dataset/test/X_test.txt")
test_Y<-read.table("UCI HAR Dataset/test/y_test.txt")
col<- read.table("UCI HAR Dataset/features.txt")
col_names <- readLines("UCI HAR Dataset/features.txt")
colnames(train_X)<-make.names(col_names)
colnames(test_X)<-make.names(col_names)
colnames(train_Y)<-"activity"
colnames(test_Y)<-"activity"


temp5 <- read.table("UCI HAR Dataset/test/subject_test.txt")
temp6 <- read.table("UCI HAR Dataset/train/subject_train.txt") 
colnames(temp5)<-"subject"
colnames(temp6)<-"subject"

X <- cbind(test_Y, test_X) 
Y <- cbind(train_Y, train_X) 
train <- cbind( temp6, Y)
test<- cbind(temp5,X)
train <- transform(train, subjectID = factor(subject), activityID = factor(activity))
test <- transform(test, subjectID = factor(subject), activityID = factor(activity))
train$partition = "train"
test$partition = "test"

data <- rbind(train,test)

train_final<-cbind(train_Y,train_X)
test_final<-cbind(test_Y,test_X)
test_train_x<-rbind(train_X,test_X)
final_data<-rbind(train_final,test_final)
final_data$activity<-factor(final_data$activity)
levels(final_data$activity) <- c("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
table(final_data$activity)

act<-1:6
#colnames(final_data)

dim(final_data)

cat("Number of duplicated rows:", sum(duplicated(final_data)))
cat("Number of missing values:", sum(is.na(final_data)))

# Create contingency table of activity counts
activity_table <- table(final_data$activity)

# Convert to data frame
activity_counts <- data.frame(activity = names(activity_table),
                              count = as.numeric(activity_table))

ggplot(activity_counts, aes(x = activity, y = count, fill = activity)) +
  geom_bar(stat = "identity") +
  xlab("Activity") +
  ylab("Count") +
  ggtitle("Count of Activities")

levels(data$activityID) <- c("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")

par(mfrow=c(1,2))

#shows the distribution of subjects across different activities
qplot(data= data,x= subjectID,fill= activityID) + theme(plot.margin= unit(c(1,1,1,1),"cm"))

#shows the distribution of subjects across train and test partitions
qplot(data = data, x = subjectID, fill = partition) + theme(plot.margin= unit(c(1,1,1,1),"cm"))
par(mfrow=c(1,1))

colnames(final_data)[202] = "tBodyAccMagmean"

ggplot(final_data,
       aes(x =tBodyAccMagmean, group = activity, fill = activity)) +
  geom_density(alpha = .5) + 
  annotate('text', x = -.8, y = 25, label = "Stationary activities") + 
  annotate('text', x = -.0, y = 5, label = "Moving activities")

##########

ggplot(final_data,
       aes(y = activity, x = tBodyAccMagmean, group = activity, fill = activity)) +
  geom_boxplot(show.legend = FALSE)

#########
colnames(final_data)[42] = "tgravityaccx"
colnames(final_data)[43] = "tgravityaccy"
colnames(final_data)[44] = "tgravityaccz"

for (coor in c('tgravityaccx', 'tgravityaccy', 'tgravityaccz')) {
  print(
    ggplot(final_data,
           aes_string(y = 'activity', x = coor, group = 'activity', fill = 'activity')) + 
      geom_boxplot(show.legend = FALSE)
  )
}

#Hierarchical Clustering
sub1 <- subset(data, subjectID == 1)

distanceMatrix <- dist(sub1[, -c(562:564)])
hclustering <- hclust(distanceMatrix,method='complete')
dend <- as.dendrogram(hclustering)
dend <- color_branches(dend, k = 6)
plot(dend)

#k-means clustering is a more commonly used method for clustering large datasets like the HAR dataset,
#as it is computationally efficient and can handle a large number of observations and variables. 
#K-means clustering is also easier to interpret, as it produces hard clusters that assign each observation 
#to exactly one cluster. However, k-means clustering assumes that the clusters are spherical, equally sized,
#and that the variance is the same across all dimensions.




# Define the colors for each activity based on the cluster assignments
activity_colors <- c("red", "green", "blue", "purple", "orange", "gray")
names(activity_colors) <- unique(dend$labels)

# Color the branches of the dendrogram based on 6 clusters and add labels
colored_dendrogram <- color_branches(dend, k = 6)
labels_colors <- activity_colors[labels(colored_dendrogram)]

# Plot the dendrogram with a legend
plot(colored_dendrogram, main = "Activity Clusters for Subject 1")
legend("topright", legend = names(activity_colors),
       fill = activity_colors, border = NA)

# Add labels to each activity on the plot
labels_text <- paste0(sub1$activityID, " - ", sub1$activityName)
labels_text <- labels_text[colored_dendrogram$labels]
labels(colored_dendrogram) <- labels_text
# Check the labels to make sure they match the activity IDs
unique(labels_text)

















# Load the HAR dataset
library(foreign)
data<- data[,-1]
# Extract the predictor variables
predictors <- data[, 1:562]

# Set the number of clusters
k <- 6

# Perform k-means clustering
set.seed(123)
kmeans_model <- kmeans(predictors, k)
#kmeans function to perform k-means clustering, passing in the predictor variables and the number of clusters as arguments. 
#We also set a random seed for reproducibility.

# Create a scatterplot matrix of the first six variables, colored by cluster
library(GGally)
ggpairs(data[ ,c(1,560:562)], aes(color = as.factor(kmeans_model$cluster)))
#We then create several plots to visualize the clustering results. 
#The first plot is a scatterplot matrix of the first six variables, colored by cluster. 
#This can help us see how well the clusters are separated in the data space.
#We use the ggpairs function from the GGally package to create this plot.

# Create a barplot of cluster sizes
library(ggplot2)
ggplot(data.frame(cluster = as.factor(kmeans_model$cluster))) + 
  geom_bar(aes(x = cluster, fill = cluster)) + 
  labs(title = "Cluster Sizes")
#The second plot is a barplot of cluster sizes, which can help us see how balanced the clusters are. 
#We use the ggplot2 package to create this plot.

# Create a heatmap of cluster centers

install.packages("pheatmap")
library(pheatmap)
pheatmap(kmeans_model$centers, 
         main = "Cluster Centers", 
         cluster_colnames = FALSE,
         cluster_rownames = FALSE)
#The third plot is a heatmap of the cluster centers, which can help us see the patterns of variable values that define each cluster. 
#We use the pheatmap function from the pheatmap package to create this plot.




# Choose two variables to plot
var1 <- 1
var2 <- 2

# Create a dataframe with the chosen variables and cluster assignments
df <- data[, c(var1, var2)]
df$cluster <- as.factor(kmeans_model$cluster)

# Create scatter plot with color-coded clusters
library(ggplot2)
ggplot(df, aes_string(x = names(df)[1], y = names(df)[2], color = "cluster")) +
  geom_point() +
  labs(title = paste0("Scatter Plot of Variables ", var1, " and ", var2),
       x = names(df)[1], y = names(df)[2])








numPredictors = ncol(data) - 5
dataSd = colwise(sd)(data[, 1:numPredictors])
dataSd$stat = "Predictor Variable Standard Deviation"
dataMean = colwise(mean)(data[, 1:numPredictors])
dataMean$stat = "Predictor Variable Mean"
temp = melt(rbind(dataMean, dataSd), c("stat"))
qplot(data = temp, x = value, binwidth = 0.025) + facet_wrap(~stat, ncol = 1)





qwe <- training_data[,c(42,53,50,7,559,562)]
colnames(qwe)

colnames(test_data)
asd <- test_data[,c(42,53,50,7,559,562)]
colnames(asd)








# correlation

training_data <- cbind(train_X, train_Y)
test_data <-cbind(test_X, test_Y)
test_data$activity<-as.factor(test_data$activity)
training_data$activity <- as.factor(training_data$activity)

mydata.cor = cor(train_X)
heatmap(x = mydata.cor,  symm = TRUE)




# Calculate the correlation matrix
cor_mat <- cor(training_data[, -ncol(training_data)])

library(corrplot)

#corrplot(cor_mat, method = "color")

# Select the variables with a correlation coefficient above a certain threshold (e.g., 0.7)
high_cor_vars <- findCorrelation(cor_mat, cutoff = 0.5)

# Remove the highly correlated variables from the data set
training_data_no_cor <- training_data[, -high_cor_vars]
colnames(training_data_no_cor)
# Remove the highly correlated variables from the data set
test_data_no_cor <- test_data[, -high_cor_vars]




















##Random forest feature selection
set.seed(45)
rf <- randomForest(activity ~ ., data = training_data, importance = TRUE, ntree = 500)
varImpPlot(rf)
#names(training_data_no_cor)
topFeatures <- names(sort(rf$importance[,1], decreasing = TRUE)[1:10])
topFeatures <- c(topFeatures, "activity") # Add "activity" to the list of top features


set.seed(45)
newRf <- randomForest(activity ~ ., data = training_data[, topFeatures], ntree = 500)
#names(training_data_no_cor[, topFeatures])

predictions <- predict(newRf, test_data[, topFeatures])
confusionMatrix(predictions, test_data$activity)


svm_model <- svm(activity ~ ., data = training_data[, topFeatures], kernel = "linear")
predictions <- predict(svm_model, test_data[, topFeatures])
confusionMatrix(predictions, test_data$activity)


















###############################Lasso 92.6

x <- model.matrix(activity ~ .,data=training_data_no_cor)[, -1]
y=as.factor(training_data_no_cor$activity)

grid=10^seq(10,-2,length=100)

cvfit = cv.glmnet(x,y,type.measure="class",alpha=1,family="multinomial", lambda=grid, type.multinomial="grouped")
coef=coef(cvfit, s = "lambda.min")
#fit <- glmnet(as.matrix(training_data[, -562]), training_data$activity, alpha = 1, family = "multinomial", lambda = lambda.min)

#coef

par(mar=c(1,1,1,1))
par(mfrow=c(6,1))
plot(coef[[1]], xlab="WALKING", ylab= "Predictors")
plot(coef[[2]], xlab="WALKING_UPSTAIRS", ylab="Predictors") 
plot(coef[[3]], xlab= "WALKING_DOWNSTAIRS",  ylab= "Predictors")
plot(coef[[4]], xlab= "SITTING",  ylab= "Predictors")
plot(coef[[5]], xlab="STANDING", ylab= "Predictors")
plot(coef[[6]], xlab="LAYING", ylab="Predictors")


#legend("topright", legend=colnames(coef_matrix), col=1:6, lty=1) ##################


x_t <- model.matrix(activity ~ .,data=test_data_no_cor)[, -1]
y_t=as.factor(test_data_no_cor$activity)

pred_class <- predict(cvfit, newx = x_t, s = 0, type="class" )
tab=table(pred_class, y_t)
acc <- sum(tab[row(tab) == col(tab)]) / sum(tab)
acc

pred_class<-as.factor(pred_class)
confusion_matrix <- confusionMatrix(pred_class, test_data$activity)
print(confusion_matrix)
#######
#fitting a multinomial logistic regression model using the glmnet package and evaluating the performance of the model on a test set.

#First, the code creates a design matrix x and a response vector y from the training data training_data_no_cor, where activity is the outcome variable and the remaining variables are predictors. The model.matrix() function is used to create the design matrix and the as.factor() function is used to convert the outcome variable to a factor.

#Next, the code sets up a grid of values for the regularization parameter lambda and uses the cv.glmnet() function to perform cross-validation and select the optimal value of lambda based on classification accuracy (type.measure="class"). The alpha parameter specifies the elastic-net mixing parameter (1 corresponds to Lasso regression), and type.multinomial="grouped" specifies that the model uses grouped multinomial regression.

#After selecting the optimal value of lambda, the code extracts the estimated coefficients at that value using the coef() function and plots them separately for each activity level.

#The code then creates a design matrix x_t and a response vector y_t from the test data test_data_no_cor, and uses the predict() function to obtain class predictions for the test data using the fitted model at the selected value of lambda. The resulting predicted classes are compared to the true classes in the test data using a confusion matrix and the classification accuracy is calculated.

#Finally, the code uses the confusionMatrix() function from the caret package to create a more detailed confusion matrix and print it to the console.
################


#svm
library(e1071)

# Train the SVM model
svm_model <- svm(activity ~ ., data = training_data_no_cor, kernel = "linear")

# Make predictions on the test set
svm_pred <- predict(svm_model, newdata = test_data_no_cor)

# Calculate accuracy and print confusion matrix
svm_acc <- mean(svm_pred == test_data_no_cor$activity)
svm_acc
svm_conf_matrix <- table(svm_pred, test_data_no_cor$activity)
print(svm_conf_matrix)
confusion_matrix_svm <- confusionMatrix(svm_pred, test_data_no_cor$activity)
print(confusion_matrix_svm)


# Convert predicted and true labels to factors with the same levels
pred_svm <- factor(svm_pred, levels = levels(test_data_no_cor$activity))
y_test <- test_data_no_cor$activity
y_test <- factor(y_test, levels = levels(test_data_no_cor$activity))

# Calculate precision and recall for each class
precision_svm <- numeric(length(levels(test_data_no_cor$activity)))
recall_svm <- numeric(length(levels(test_data_no_cor$activity)))
for (i in seq_along(precision_rf)) {
  tp <- sum(pred_svm == levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  fp <- sum(pred_svm == levels(test_data_no_cor$activity)[i] & y_test != levels(test_data_no_cor$activity)[i])
  fn <- sum(pred_svm != levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  precision_svm[i] <- tp / (tp + fp)
  recall_svm[i] <- tp / (tp + fn)
}

# Calculate F1-score for each class
f1_svm <- 2 * precision_svm * recall_svm / (precision_svm + recall_svm)

## Print the results
print(paste("Precision for each class:"))
print(precision_svm)
print(paste("recall for each class:"))
print(recall_svm)
print(paste("F1-score for each class:"))
print(f1_svm)

#Decision Tree

library(rpart)
library(rpart.plot)
# Fit decision tree model
fit <- rpart(activity ~ ., data = training_data_no_cor, method = "class")
rpart.plot(fit)

rpart.plot(fit, extra = 101, box.palette = "Reds",
           branch.lty = 3, under.cex = 0.8,
           varlen = 0, shadow.col = "gray",
            cex = 0.8,
           main = "Decision Tree for HAR Dataset")
legend("topright", legend = c("Class: C1", "Class: C2"),
       col = c("red", "blue"), pch = 20, cex = 0.8)

# Create prediction function
predict_class <- function(fit, newdata) {
  predict(fit, newdata, type = "class")
}

# Create design matrix and response vector for test data
x_test <- test_data_no_cor[, -ncol(test_data_no_cor)]
y_test <- test_data_no_cor$activity

# Predict class labels for test data using fitted model
pred_class <- predict_class(fit, newdata = x_test)

# Calculate classification accuracy
acc <- mean(pred_class == y_test)
acc

library(rpart)

# Fit the decision tree model
fit_dt <- rpart(activity ~ ., data = training_data_no_cor)

# Make predictions on the test data
pred_dt <- predict(fit_dt, newdata = test_data_no_cor, type = "class")

# Generate the confusion matrix
confusion_matrix_dt <- confusionMatrix(pred_dt, test_data_no_cor$activity)
print(confusion_matrix_dt)

mcnemar.test(table(pred_dt,svm_pred))

# Convert predicted and true labels to factors with the same levels
pred_dt <- factor(pred_dt, levels = levels(test_data_no_cor$activity))
y_test <- factor(y_test, levels = levels(test_data_no_cor$activity))

# Calculate precision and recall for each class
precision_dt <- numeric(length(levels(test_data_no_cor$activity)))
recall_dt <- numeric(length(levels(test_data_no_cor$activity)))
for (i in seq_along(precision_dt)) {
  tp <- sum(pred_dt == levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  fp <- sum(pred_dt == levels(test_data_no_cor$activity)[i] & y_test != levels(test_data_no_cor$activity)[i])
  fn <- sum(pred_dt != levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  precision_dt[i] <- tp / (tp + fp)
  recall_dt[i] <- tp / (tp + fn)
}

# Calculate F1-score for each class
f1_dt <- 2 * precision_dt * recall_dt / (precision_dt + recall_dt)

## Print the results
print(paste("Precision for each class:"))
print(precision_dt)
print(paste("recall for each class:"))
print(recall_dt)
print(paste("F1-score for each class:"))
print(f1_dt)



#Random Forest

library(randomForest)

# Fit random forest model
fit <- randomForest(activity ~ ., data = training_data_no_cor, ntree = 500, importance = TRUE)

# Create prediction function
predict_class <- function(fit, newdata) {
  predict(fit, newdata, type = "class")
}

# Create design matrix and response vector for test data
x_test <- test_data_no_cor[, -ncol(test_data_no_cor)]
y_test <- test_data_no_cor$activity

# Predict class labels for test data using fitted model
pred_class <- predict_class(fit, newdata = x_test)

# Calculate classification accuracy
acc <- mean(pred_class == y_test)

acc

library(randomForest)

# Fit the random forest model
fit_rf <- randomForest(activity ~ ., data = training_data_no_cor)

# Make predictions on the test data
pred_rf <- predict(fit_rf, newdata = test_data_no_cor)

# Generate the confusion matrix
confusion_matrix_rf <- confusionMatrix(pred_rf, test_data_no_cor$activity)
print(confusion_matrix_rf)
########################

# Convert predicted and true labels to factors with the same levels
pred_rf <- factor(pred_rf, levels = levels(test_data_no_cor$activity))
y_test <- factor(y_test, levels = levels(test_data_no_cor$activity))

# Calculate precision and recall for each class
precision_rf <- numeric(length(levels(test_data_no_cor$activity)))
recall_rf <- numeric(length(levels(test_data_no_cor$activity)))
for (i in seq_along(precision_rf)) {
  tp <- sum(pred_rf == levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  fp <- sum(pred_rf == levels(test_data_no_cor$activity)[i] & y_test != levels(test_data_no_cor$activity)[i])
  fn <- sum(pred_rf != levels(test_data_no_cor$activity)[i] & y_test == levels(test_data_no_cor$activity)[i])
  precision_rf[i] <- tp / (tp + fp)
  recall_rf[i] <- tp / (tp + fn)
}

# Calculate F1-score for each class
f1_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)

## Print the results
print(paste("Precision for each class:"))
print(precision_rf)
print(paste("recall for each class:"))
print(recall_rf)
print(paste("F1-score for each class:"))
print(f1_rf)
