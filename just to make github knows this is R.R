
if (!require("readxl")) install.packages("readxl")

seven <- read_excel("已標記顧客表 _201907.xlsx")
eight <- read_excel("已標記顧客表 _201908.xlsx")
nine <- read_excel("已標記顧客表 _201909.xlsx")

total <- rbind(seven,eight,nine)

# 看男女分配數
value_counts <- table(total$性別)




