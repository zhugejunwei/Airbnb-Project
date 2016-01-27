library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(pROC)

train <- read.csv('train_users_2.csv', stringsAsFactors = FALSE)

labels <- train$country_destination
binaryLabel <- rep(1, dim(train)[1])
binaryLabel[which(labels == 'NDF')] <- 0
labelInd <- which(colnames(train) == "country_destination")
train <- train[, -labelInd]

featuresToDummies <- c('gender', 'signup_method', 'signup_flow', 'language', 
                       'affiliate_channel', 'affiliate_provider', 
                       'first_affiliate_tracked', 'signup_app', 'first_device_type', 
                       'first_browser')
dummies <- dummyVars(~ gender + signup_method + signup_flow + language + 
                       affiliate_channel + affiliate_provider + 
                       first_affiliate_tracked + signup_app + 
                       first_device_type + first_browser, data = train)
featuresInDummies <- as.data.frame(predict(dummies, newdata = train))
# Need to change the column names, otherwise, column names with special character
# such as -, (), will cause error when calling the glm function
colnames(featuresInDummies) <- gsub(" ", "", colnames(featuresInDummies))
colnames(featuresInDummies) <- gsub("-", "", colnames(featuresInDummies))
colnames(featuresInDummies) <- gsub("(Other)", "Other", 
                                    colnames(featuresInDummies),
                                    fixed = TRUE)
colnames(featuresInDummies) <- gsub("/", "", 
                                    colnames(featuresInDummies))

train <- cbind(train[,-c(which(colnames(train) %in% featuresToDummies))], 
               featuresInDummies)

train <- train[, - which(colnames(train) == 'date_first_booking')]
dac <- as.data.frame(str_split_fixed(train$date_account_created, "-", 3))
train <- train[, - which(colnames(train) == 'date_account_created')]
train$dacYear <- dac[, 1]
train$dacMonth <- dac[, 2]
train$dacDay <- dac[, 3]
train$tfaYear <- substring(as.character(train$timestamp_first_active), 1, 4)
train$tfaMonth <- substring(as.character(train$timestamp_first_active), 5, 6)
train$tfaDay <- substring(as.character(train$timestamp_first_active), 7, 8)
train <- train[, - which(colnames(train) == 'timestamp_first_active')]

train[which(train$age < 14 | train$age > 90), 'age'] <- -1
train[which(is.na(train)), 'age'] <- -1
train <- cbind(train, binaryLabel)

set.seed(1)
modelTrainInd <- runif(dim(train)[1])
modelTrain <- train[which(modelTrainInd < 0.7), ]
modelTest <- train[which(modelTrainInd >= 0.7), ]
binaryLabelTrain <- binaryLabel[which(modelTrainInd < 0.7)]
binaryLabelTest <- binaryLabel[which(modelTrainInd >= 0.7)]


includeCols <- NULL
for (i in c(2:dim(modelTrain)[2])){
  if(length(unique(modelTrain[, i])) > 1) {
    includeCols <- c(includeCols, i)
  }
}

formula <- "binaryLabelTrain ~ "
for (j in 1:(length(includeCols) - 1)){
  formula <- paste(formula, colnames(modelTrain)[includeCols[j]], "+")
}
formula <- paste(formula, colnames(modelTrain)[includeCols[j + 1]])

logitBinaryLabel <- glm(formula, data = modelTrain, family = binomial)
predBinaryLabel <- predict(logitBinaryLabel, newdata = modelTest, type = 'response')
head(predBinaryLabel)
plot.roc(binaryLabelTest, predBinaryLabel)