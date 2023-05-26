install.packages("mlbench")
install.packages("randomForest")
install.packages("e1071")
install.packages("kernlab")
install.packages("caret")
library("caret")
library("mlbench")
library("randomForest")


data(Satellite)
map <- Satellite
map_index <- createDataPartition(map$classes, p=0.80, list=FALSE)
training <- map[map_index,]
test <- map[-map_index,]
set.seed(7)

#Confusion Matrix
rf <- train(classes~., data=training, method="rf")
predicoes.rf <- predict(rf, test)
confusionMatrix(predicoes.rf, test$classes)

Prediction            red soil cotton crop grey soil damp grey soil vegetation stubble very damp grey soil
  red soil                 300           0         0              1                 13                   0
  cotton crop                0         138         0              1                  4                   0
  grey soil                  5           0       261             23                  0                   7
  damp grey soil             0           1         5             72                  0                  14
  vegetation stubble         1           0         1              1                117                   3
  very damp grey soil        0           1         4             27                  7                 277

Overall Statistics
                                          
               Accuracy : 0.9073          
                 95% CI : (0.8901, 0.9226)
    No Information Rate : 0.2383          
    P-Value [Acc > NIR] : < 2.2e-16       


#SVM
svm <- train(classes~., data=training, method="svmRadial")
predicoes.svm <- predict(svm, test)
confusionMatrix(predicoes.svm, test$classes)

Confusion Matrix and Statistics

                     Reference
Prediction            red soil cotton crop grey soil damp grey soil vegetation stubble very damp grey soil
  red soil                 300           0         0              0                  5                   0
  cotton crop                0         137         0              1                  5                   0
  grey soil                  4           0       263             23                  0                  13
  damp grey soil             0           1         6             68                  0                  23
  vegetation stubble         2           2         1              5                123                   1
  very damp grey soil        0           0         1             28                  8                 264

Overall Statistics
                                          
               Accuracy : 0.8995          
                 95% CI : (0.8818, 0.9154)
    No Information Rate : 0.2383          
    P-Value [Acc > NIR] : < 2.2e-16    

#RNA
rna <- train(classes~., data=training, method="nnet", trace=FALSE)
predicoes.rna <- predict(rna, test)
confusionMatrix(predicoes.rna, test$classes)

Confusion Matrix and Statistics

                     Reference
Prediction            red soil cotton crop grey soil damp grey soil vegetation stubble very damp grey soil
  red soil                 296         118         2              2                 23                   0
  cotton crop                0           0         0              0                  0                   0
  grey soil                  0           0         0              0                  0                   0
  damp grey soil             0           0         0              0                  0                   0
  vegetation stubble         4          22         1              7                 83                  10
  very damp grey soil        6           0       268            116                 35                 291

Overall Statistics
                                          
               Accuracy : 0.5218          
                 95% CI : (0.4941, 0.5494)
    No Information Rate : 0.2383          
    P-Value [Acc > NIR] : < 2.2e-16  
    
    
#Portanto, o método que teve mais eficiencia foi o Random Forest
print(rf)
print(svm)
print(rna)

#Agora realizando o teste com 100% da base

final_model <- randomForest(type="C-svc", classes~., data=map, kernel="rbfdot",C=1.0, kpar=list(sigma=0.01173596))
final_predict.rf <- predict(final_model, map)
confusionMatrix(final_predict.rf, map$classes)
saveRDS(final_model, "map_rf.rds")

#Como Resultado final temos um acuracia igual 1 o que é uma precisão excelente, portanto, o método Random Forest se mostra o mais eficiente em todos os cenários

