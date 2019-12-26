#安裝catboost------------------------------------------------------------------------------------------------------------
install.packages('devtools')
#如果有warning顯示要下載Rtools的話，就照他的網址下載然後安裝
#我下載的是Rtools35
#安裝的時候記得把 Add rtools to system PATH打勾

install.packages('rlang') #更新rlang

# 請去這邊 https://catboost.ai/docs/installation/r-installation-local-copy-installation-windows.html 按照指示安裝catboost

setwd("C:/Users/willy/CatBoostRepository/catboost/catboost/R-package") #default path for my catboost

library(devtools)
devtools::build()
devtools::install()
#------------------------------------------------------------------------------------------------------------------------


#https://github.com/catboost/tutorials/blob/master/r_tutorial.ipynb
library(catboost)

setwd("C:/Users/willy/OneDrive/桌面/ntu mba/grade two semester 1/business analytics/期末專案資料")
total<-read.csv('unifed.csv')
total_alter<-total

total_alter$結果標記<-as.character(total_alter$結果標記)
class(total_alter$結果標記)

total_alter[total_alter$結果標記  == '續用', '結果標記']<-'0'
total_alter[total_alter$結果標記  == '流失', '結果標記']<-'1'
total_alter[total_alter$結果標記  == '不動作', '結果標記']<-'0'
total_alter$結果標記<-as.integer((total_alter$結果標記))


#資料前處理 !!重要!! catboost只吃某幾種變數種類
#把所有文字行資料都轉成factor
total_alter[sapply(total_alter, is.character)] <- lapply(total_alter[sapply(total_alter, is.character)], 
                                           as.factor)


#去掉不合理的變數
total_alter$X <- NULL


# 切分train/val
total_train <- total_alter[(total_alter$統計年月 == '201907' | total_alter$統計年月 == '201908'),]
total_val <-  total_alter[total_alter$統計年月 == '201909',]


#用7、8月的資料訓練
train_total_data <- total_train[, -18]
train_total_label <- total_train$結果標記


# 9月資料當做測試
test_data <- total_val[,-18]
test_label <- total_val$結果標記



# #去除幾個我認為用不到的變數: 統計年月
# temp <- temp[,-1]
# nine <- nine[,-1]


#這邊開始train/validation split
# 把七、八月的資料 按照0.75 0.25 隨機抽樣
smp_size <- floor(0.75 * nrow(train_total_data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train_total_data)), size = smp_size)

train_data <- train_total_data[train_ind, ]
train_label <- train_total_label[train_ind]
val_data <- train_total_data[-train_ind, ]
val_label <-train_total_label[-train_ind]

# sapply(train_label, class)


# Catboost modeling

#參數
fit_params <- list(iterations = 1000,
                   thread_count = 10,
                   loss_function = 'Logloss',
                   #ignored_features = c(4,9),
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.1,
                   l2_leaf_reg = 3.5,
                   train_dir = 'train_dir',
                   od_wait = 50,
                   logging_level = 'Silent')

#看哪些變數是factor(categorical)
#sapply(train_data, is.factor)

train_pool <- catboost.load_pool(data=train_data, label = as.numeric(unlist(train_label))  )

val_pool <- catboost.load_pool(data=val_data, label = as.numeric(unlist(val_label)))

test_pool <- catboost.load_pool(data=test_data, label = as.numeric(unlist(test_label)))


model <- catboost.train(train_pool, val_pool, fit_params)


#------------------------------Logloss from test set----------------------------------


prediction <- catboost.predict(model, test_pool, prediction_type = 'Probability')
cat("Sample predictions: ", sample(prediction, 5), "\n")


LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

LogLoss(test_label, prediction)


#--------------prediction (VALIDATION ACCURACY)----------------



calc_accuracy <- function(prediction, expected) {
  labels <- ifelse(prediction > 0.5, 1, 0)
  accuracy <- sum(labels == expected) / length(labels)
  return(accuracy)
}

prediction <- catboost.predict(model, val_pool, prediction_type = 'Probability')
cat("Sample predictions: ", sample(prediction, 5), "\n")


labels <- catboost.predict(model, val_pool, prediction_type = 'Class')
table(labels, as.numeric(unlist(val_label)))

# works properly only for Logloss
accuracy <- calc_accuracy(prediction, val_label)
cat("\nAccuracy: ", accuracy, "\n")

# feature splits importances (not finished)

cat("\nFeature importances", "\n")
catboost.get_feature_importance(model, train_pool)

cat("\nTree count: ", model$tree_count, "\n")

#----------------prediction (TEST ACCURACY)-----------------------

calc_accuracy <- function(prediction, expected) {
  labels <- ifelse(prediction > 0.5, 1, 0)
  accuracy <- sum(labels == expected) / length(labels)
  return(accuracy)
}

prediction <- catboost.predict(model, test_pool, prediction_type = 'Probability')
cat("Sample predictions: ", sample(prediction, 5), "\n")


pred_labels <- catboost.predict(model, test_pool, prediction_type = 'Class')
table(pred_labels, as.numeric(unlist(test_label)))



# works properly only for Logloss
accuracy <- calc_accuracy(prediction, test_label)
cat("\nAccuracy: ", accuracy, "\n")

# feature splits importances (not finished)

cat("\nFeature importances", "\n")
catboost.get_feature_importance(model, train_pool)

cat("\nTree count: ", model$tree_count, "\n")
