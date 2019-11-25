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
seven <- read_excel("已標記顧客表 _201907.xlsx")
eight <- read_excel("已標記顧客表 _201908.xlsx")
nine <- read_excel("已標記顧客表 _201909.xlsx")





# eight[eight$編號 %in% nine$編號 == F, ]
temp <- rbind(eight[eight$編號 %in% nine$編號 == F, ], nine) # 8,9月·不重複
temp_2 <- rbind(seven[seven$編號 %in% temp$編號 == F,], temp) # 7,8,9月·不重複

#資料前處理 !!重要!! catboost只吃某幾種變數種類

#去除幾個我認為用不到的變數: 統計年月
temp <- temp[,-1]
nine <- nine[,-1]


#把"結果標記"之續用、不動作轉成0,流失轉成1
temp$結果標記[temp$結果標記=="不動作"] = "0"
temp$結果標記[temp$結果標記=="續用"] = "0"
temp$結果標記[temp$結果標記=="流失"] = "1"
nine$結果標記[nine$結果標記=="不動作"] = "0"
nine$結果標記[nine$結果標記=="續用"] = "0"
nine$結果標記[nine$結果標記=="流失"] = "1"

temp$結果標記 <- as.numeric(temp$結果標記)
nine$結果標記 <- as.numeric(nine$結果標記)


#把所有文字行資料都轉成factor
temp[sapply(temp, is.character)] <- lapply(temp[sapply(temp, is.character)], 
                                       as.factor)
# sapply(temp, class)
nine[sapply(nine, is.character)] <- lapply(nine[sapply(nine, is.character)], 
                                           as.factor)
# sapply(nine, class)






#用7、8月的資料訓練
train_total_data <- temp[,-ncol(temp)]
train_total_label <- temp[,ncol(temp)]


# 9月資料當做測試
test_data <- nine[,-ncol(temp)]
test_label <- nine[,ncol(temp)]


#這邊開始train/validation split
smp_size <- floor(0.75 * nrow(train_total_data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train_total_data)), size = smp_size)

train_data <- train_total_data[train_ind, ]
train_label <- train_total_label[train_ind,]
val_data <- train_total_data[-train_ind, ]
val_label <-train_total_label[-train_ind,]

sapply(train_label, class)


# Catboost modeling

#參數
fit_params <- list(iterations = 100,
                   thread_count = 10,
                   loss_function = 'Logloss',
                   #ignored_features = c(4,9),
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.1,
                   l2_leaf_reg = 3.5,
                   train_dir = 'train_dir',
                   logging_level = 'Silent')

#看哪些變數是factor(categorical)
#sapply(train_data, is.factor)

train_pool <- catboost.load_pool(data=train_data, label = as.numeric(unlist(train_label))  )

val_pool <- catboost.load_pool(data=val_data, label = as.numeric(unlist(val_label)))

test_pool <- catboost.load_pool(data=test_data, label = as.numeric(unlist(test_label)))


fit_params <- list(iterations = 1000,
                   thread_count = 10,
                   loss_function = 'Logloss',
                   #ignored_features = c(4,9),
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.03,
                   l2_leaf_reg = 3.5,
                   train_dir = 'train_dir',
                   #logging_level = 'Silent'
                   )

model <- catboost.train(train_pool, val_pool, fit_params)



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


labels <- catboost.predict(model, test_pool, prediction_type = 'Class')
table(labels, as.numeric(unlist(test_label)))

# works properly only for Logloss
accuracy <- calc_accuracy(prediction, test_label)
cat("\nAccuracy: ", accuracy, "\n")

# feature splits importances (not finished)

cat("\nFeature importances", "\n")
catboost.get_feature_importance(model, train_pool)

cat("\nTree count: ", model$tree_count, "\n")
