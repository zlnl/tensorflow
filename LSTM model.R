# Install Libraries
# Readxl
library(readxl)

# Core tidyverse
library(tidyverse)
library(glue)
library(forcats)

# Time series
library(zoo)
library(timetk)
library(tidyquant)
library(tibbletime)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
install.packages("reticulate")
library(reticulate)
use_python("/usr/local/bin/python")
use_virtualenv(virtualenv = NULL, required = FALSE)
use_virtualenv("myenv")

library(keras)
library(tensorflow)
install_tensorflow(method="conda")



# Import data
vixft <- read_xlsx("VIXindex_grid1_otvxi4iq.xlsx") %>%
  as_tibble(read_xlsx("VIXindex_grid1_otvxi4iq.xlsx")) %>%
  mutate(index = as_date(Dates))
vixft <- vixft[-1] %>% as_tbl_time(index = index)


# ---------https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/----------

# *******************************************Data Analysis*****************************************************
# ACF
##PX_LAST: closing price
tidy_acf <- function(vixft, PX_LAST, lags = 0:20) {
  value_expr <- enquo(PX_LAST)
  acf_values <- vixft %>%
    pull(PX_LAST) %>%
    acf(lag.max = tail(lags, 1), plot = FALSE) %>% .$acf %>%
    .[,,1]
  ret <- tibble(acf = acf_values) %>% rowid_to_column(var = "lag") %>% mutate(lag = lag - 1) %>% filter(lag %in% lags)
  return(ret)
}
max_lag <- 12 * 50
vixft %>%
  tidy_acf(PX_LAST, lags = 0:max_lag)



# Check whether there is high autocorrelation lag exists beyond 10 years
vixft %>%
  tidy_acf(PX_LAST, lags = 0:max_lag) %>%
  ggplot(aes(lag, acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) + geom_vline(xintercept = 120, size = 3, color = palette_light()[[2]]) + annotate("text", label = "10 Year Mark", x = 130, y = 0.8,
                                                                                                                                                          color = palette_light()[[2]], size = 6, hjust = 0) + theme_tq() +
  labs(title = "ACF: VIXfutures")

optimal_lag_setting <- vixft %>% tidy_acf(PX_LAST, lags = 115:135) %>% filter(acf == max(acf)) %>% pull(lag)




# Backtesting with crossvalidation
periods_train <- 12 * 50
periods_test  <- 12 * 10
skip_span     <- 12 * 20
rolling_origin_resamples <- rolling_origin(vixft,
                                           initial = periods_train, assess = periods_test, cumulative = FALSE,
                                           skip = skip_span)
rolling_origin_resamples




# *******************************************Modeling The Keras Stateful LSTM Model*****************************************************
# Single LSTM
split <- rolling_origin_resamples$splits[[11]] 
split_id <- rolling_origin_resamples$id[[11]]



# Data setup
df_trn <- training(split) 
df_tst <- testing(split)
df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
) %>%
  as_tbl_time(index = index)
df




# Preprocessing with recipes (The LSTM algorithm requires the input data to be centered and scaled)
rec_obj <- recipe(PX_LAST ~ ., df) %>% step_sqrt(PX_LAST) %>% step_center(PX_LAST) %>% step_scale(PX_LAST) %>%
  prep()
df_processed_tbl <- bake(rec_obj, df)
df_processed_tbl



# Capture the center/scale history
center_history <- rec_obj$steps[[2]]$means["PX_LAST"] 
scale_history <- rec_obj$steps[[3]]$sds["PX_LAST"]
c("center" = center_history, "scale" = scale_history)




# LSTM plan
# Model inputs
lag_setting <- 120 # = nrow(df_tst) 
batch_size <- 40
train_length <- 440
tsteps <-1
epochs <- 300




# 2D And 3D Train/Test Arrays
# Training Set
lag_train_tbl <- df_processed_tbl %>%
  mutate(PX_LAST_lag = lag(PX_LAST, n = lag_setting)) %>%
  filter(!is.na(PX_LAST_lag)) %>% 
  filter(key == "training") %>% tail(train_length)
x_train_vec <- lag_train_tbl$PX_LAST_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
y_train_vec <- lag_train_tbl$PX_LAST
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
# Testing Set
lag_test_tbl <- df_processed_tbl %>%
  mutate(
    PX_LAST_lag = lag(PX_LAST, n = lag_setting)
  ) %>% filter(!is.na(PX_LAST_lag)) %>% filter(key == "testing")

x_test_vec <- lag_test_tbl$PX_LAST_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
y_test_vec <- lag_test_tbl$PX_LAST
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))




# Building the LSTM model
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps, 1),
             batch_size = batch_size,
             return_sequences = TRUE,
             stateful = TRUE) %>% layer_lstm(units = 50, return_sequences = FALSE, stateful = TRUE) %>%
  layer_dense(units = 1)                     
model %>%
  compile(loss = 'mae', optimizer = 'adam')

model 




# Fitting LSTM
for (i in 1:epochs) {
  model %>% fit(x = x_train_arr,
                y = y_train_arr,
                batch_size = batch_size,
                epochs = 1,
                verbose = 1,
                shuffle = FALSE)
  
  model %>% reset_states() 
  cat("Epoch: ", i)
}




# Predicting using LSTM
# Make Predictions
pred_out <- model %>%
  predict(x_test_arr, batch_size = batch_size) %>% .[,1]
# Retransform values
pred_tbl <- tibble(
  index = lag_test_tbl$index,
  value = (pred_out * scale_history + center_history)^2 )
# Combine actual data with predictions 
tbl_1 <- df_trn %>% add_column(key = "actual")
tbl_2 <- df_tst %>% add_column(key = "actual")
tbl_3 <- pred_tbl %>% add_column(key = "predict")
# Create time_bind_rows() to solve dplyr issue 
time_bind_rows <- function(data_1, data_2, index) {
  index_expr <- enquo(index) 
  bind_rows(data_1, data_2) %>%
    as_tbl_time(index = !! index_expr)
}
ret <- list(tbl_1, tbl_2, tbl_3) %>% reduce(time_bind_rows, index = index) %>% arrange(key, index) %>%
  mutate(key = as_factor(key))
ret




# Assessing the performance of LSTM on a single split
calc_rmse <- function(prediction_tbl) {
  rmse_calculation <- function(data) { data %>%
      spread(key = key, PX_LAST = PX_LAST) %>% select(-index) %>% filter(!is.na(predict)) %>%
      rename(
        truth    = actual,
        estimate = predict
      ) %>%
      rmse(truth, estimate)
  }
  safe_rmse <- possibly(rmse_calculation, otherwise = NA) 
  safe_rmse(prediction_tbl)
}


