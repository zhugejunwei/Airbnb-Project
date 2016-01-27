library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(pROC)
library(Ckmeans.1d.dp)

# set wd to be kaggle folder
train <- read.csv('/Users/zhugejunwei/Documents/Semester 2/DM/Final Project/data/train_users_2.csv', stringsAsFactors = FALSE)
test <- read.csv('/Users/zhugejunwei/Documents/Semester 2/DM/Final Project/data/test_users.csv', stringsAsFactors = FALSE)
# clean the training data
labels <- train$country_destination
labelInd <- which(colnames(train) == "country_destination")
train <- train[, -labelInd]

# preds is an array of five predictions in order of the possibility,
# each prediction is a country (chr)
ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = 12
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

# combine train and test
dataAll <- rbind(train, test)
dataAll <- dataAll[, - which(colnames(dataAll) == 'date_first_booking')]
dac <- as.data.frame(str_split_fixed(dataAll$date_account_created, "-", 3))
dataAll$dacYear <- dac[, 1]
dataAll$dacMonth <- dac[, 2]
dataAll$dacDay <- dac[, 3]
dataAll$tfaYear <- substring(as.character(dataAll$timestamp_first_active[1]), 1, 4)
dataAll$tfaMonth <- substring(as.character(dataAll$timestamp_first_active[1]), 5, 6)
dataAll$tfaDay <- substring(as.character(dataAll$timestamp_first_active[1]), 7, 8)
dataAll <- dataAll[, - which(colnames(dataAll) == 'timestamp_first_active')]

dataAll[which(dataAll$age < 14 | dataAll$age > 90), 'age'] <- -1
dataAll[which(is.na(dataAll$age)), 'age'] <- -1

featuresToDummies <- c('gender', 'signup_method', 'signup_flow', 'language', 
                       'affiliate_channel', 'affiliate_provider', 
                       'first_affiliate_tracked', 'signup_app', 'first_device_type', 
                       'first_browser')
dummies <- dummyVars(~ gender + signup_method + signup_flow + language + 
                       affiliate_channel + affiliate_provider + 
                       first_affiliate_tracked + signup_app + 
                       first_device_type + first_browser, data = dataAll)
featuresInDummies <- as.data.frame(predict(dummies, newdata = dataAll))
dataAll <- cbind(dataAll[,-c(which(colnames(dataAll) %in% featuresToDummies))], featuresInDummies)

dataTrain <- dataAll[which(dataAll$id %in% train$id), ] # get the train set
set.seed(1)
modelTrainInd <- runif(dim(dataTrain)[1]) # generates random deviates
modelTrain <- dataAll[which(modelTrainInd < 0.7), ] 
modelTest <- dataAll[which(modelTrainInd >= 0.7), ]

lableTrain <- recode(labels, "'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 
                     'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 
                     'AU'=11")
modelTrainLabel <- lableTrain[which(modelTrainInd < 0.7)]


# train xgboost (multi class)
xgb<- xgboost(data = data.matrix(modelTrain[,-c(1,2)]), 
              label = modelTrainLabel, 
              max_depth = 8, 
              nround=20, 
              eval_metric = ndcg5,
              objective = "multi:softprob",
              num_class = 12,
              nthread = 3
)

# predict values in test set
predXgb <- predict(xgb, data.matrix(modelTest[,-c(1,2)]))

importance_matrix <- xgb.importance(colnames(modelTrain[,-c(1,2)]), model = xgb)
xgb.plot.importance(importance_matrix[1:10,])
