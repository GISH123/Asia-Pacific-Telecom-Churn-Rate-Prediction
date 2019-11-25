# Asia-Pacific-Telecom-Churn-Rate-Prediction\
亞太電信 顧客續用與流失率預測

顧客購買A公司商品的交易模式為每個月支付A公司商品使用費,同時綁定一個持續使用A公司商品的約定期間(簡稱:綁約期),當顧客的綁約期快到時,A公司想要預測哪些顧客會繼續綁下一個約付商品使用費(簡稱:續用率)? 哪些顧客會離開(簡稱:流失率)? 

使用R語言來建模

2019/11/22  EDA   

2019/11/25  Catboost建模  
使用前兩個月（不重複客戶）最近的資料，也就是說，如果該客戶出現在兩個月的資料，只取他最近一個月的資料代表該人
前兩個月的資料以0.75, 0.25切出train/validation
第三個月的資料當作test

/***
|              | Predictions    |               | 
| Labels       | 0              |             1 | 
| --- | --- | --- | 
| 0            | 36559          | 3306          | 
| 1            | 910            | 1990          | 
***/

Confusion matrix(Validation data):   
&nbsp;&nbsp; prediction     
labels &nbsp;0&nbsp;1   
&nbsp;0 36559  3306   
&nbsp;1   910  1990  
Accuracy:  0.9014147   


Confusion matrix(Test data):   
&nbsp;&nbsp;prediction    
labels &nbsp;0 &nbsp;1   
&nbsp;0 133420   6016   
&nbsp;1   2669   4544   
     
Accuracy:  0.940777    

這邊有個現象，Test dataset的準確度比validation set還高    
我個人猜測是因為test資料量比較大，且此模型（比較generic?)所以導致accuracy比較高   
或是哪個地方出錯才導致這種現象   
