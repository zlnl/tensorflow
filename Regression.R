library(tidyverse)
library(glue)
library(forcats)
library(timetk)
library(tidyquant)
library(tibbletime)
library(dplyr)
library(cowplot)
library(recipes)
library(rsample)
library(yardstick) 
library(TTR)
library(zoo)
library(timeSeries)
library(readxl)



# Import data
VIXindex <- read_xlsx("SPXindex_grid1_otvxi4iq.xlsx") 
vixft_output <- read_xlsx("LSTM_output.xlsx")
#vixft_output <- vixft_output[-1]



# Build normalize function
VIX_values <- VIXindex[-1]
normalize <- function(VIX_values) {
  num <- VIX_values - min(VIX_values)
  denom <- max(VIX_values) - min(VIX_values)
  return (num/denom)
}



# Normalize the data
VIX_norm <- as.data.frame(lapply(VIX_values, normalize))
head(VIX_norm)
VIXindex <- cbind(VIXindex[1], VIX_norm)



# Process data
yhat <- VIXindex[3018:3137, 2]
x <- vixft_output[1:120, 1]
x <- unlist(x)
class(yhat)
class(x)
yhat <- unlist(yhat)
reg_data <- data.frame(yhat, x)



# Build linear model 
lmodel <- lm(yhat ~ x, data = reg_data)



# Add predictions 
pred.int <- predict(lmodel, interval = "prediction")
mydata <- cbind(reg_data, pred.int) 



# Regression line + confidence intervals
library("ggplot2")
VIX_futures_predictions <- x
VIX_index <- yhat
p <- ggplot(mydata, aes(VIX_futures_predictions, VIX_index)) +
  geom_point() +
  stat_smooth(method = lm)
# Add prediction intervals
p + geom_line(aes(y = lwr), color = "black", linetype = "dashed")+
  geom_line(aes(y = upr), color = "black", linetype = "dashed")



# t-statistic and p-value
modelSummary <- summary(lmodel) 
modelCoeffs <- modelSummary$coefficients 
beta.estimate <- modelCoeffs["x", "Estimate"]  
std.error <- modelCoeffs["x", "Std. Error"]  
t_value <- beta.estimate/std.error  # calc t statistic
p_value <- 2*pt(-abs(t_value), df=nrow(reg_data)-ncol(reg_data))  
f_statistic <- lmodel$fstatistic[1]  
f <- summary(lmodel)$fstatistic  
model_p <- pf(f[1], f[2], f[3], lower=FALSE)



# Predicting linear model 
# Create Training and Test data
trainingRowIndex <- sample(1:nrow(reg_data), 0.8*nrow(reg_data)) 
trainingData <- reg_data[trainingRowIndex, ]  
testData  <- reg_data[-trainingRowIndex, ]   

# Build the model on training data -
lmMod <- lm(yhat ~ x, data=trainingData)  
yhatPred <- predict(lmMod, testData)  
summary (lmMod)  



# Prediction accuracy
actuals_preds <- data.frame(cbind(actuals=testData$yhat, predicteds=yhatPred)) 
correlation_accuracy <- cor(actuals_preds) 
head(actuals_preds)
correlation_accuracy

# Min max accuracy and MAPE
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
min_max_accuracy

mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  
mape



