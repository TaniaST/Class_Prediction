#loading needed libraries

library(caret)
library(factoextra)
library(GGally)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)

#loading the file with data
load("oliveoils.RData")

#1. Preparing data
#extracting the relevant variables and adding class labels
fulldata<-oliveoils[,c("720":"820")]
fulldata<-cbind(oliveoillabels,as.data.frame(fulldata))

head(fulldata)
dim(fulldata)

#Splitting data into three sets: training, validation and test
set.seed(200)
train.rows<-createDataPartition(y=fulldata$oliveoillabels,p=0.5,list=FALSE)
train.set<-fulldata[train.rows,]# 50% of data goes to training set
testvalid<-fulldata[-train.rows,]#the remaining 50% of data
valid.rows<-createDataPartition(y=testvalid$oliveoillabels,p=0.5,list=FALSE)
valid.set<-testvalid[valid.rows,]
test.set<-testvalid[-valid.rows,]

#checking that we've got relatively fair split between Crete/Other/Peloponese
props<-rbind(round(prop.table(table(train.set$oliveoillabels)),2),
             round(prop.table(table(valid.set$oliveoillabels)),2),
             round(prop.table(table(test.set$oliveoillabels)),2),
             round(prop.table(table(fulldata$oliveoillabels)),2))
rownames(props)<-c("Training set","Validation set","Test set", "Full set")

props

#2.Exploratory analysis

#as we have 101 variables, it's difficult to visualise and judge on the graphs.
#let's use the boxplot to see if we have any outliers

boxplot(train.set[,-1])
#we can see that quite a few variables potentially have outliers

#let's see if there is any correlation between variables:

#correlation matrix
cor(train.set[,-1])

ggcorr(train.set[,-1],palette="RdBu", label=FALSE)
#there are quite a few variables having strong positive correlation

#so, the PCA will be useful and will help to reduce dimensionality of the data
#let's have a look at standard deviations to see if we use covariance matrix or correlation matrix

sdtable<-apply(train.set[,-1],2,sd)
summary(sdtable)
#we can see from the summary that the standard deviations vary from 2354 to 5883
#so we will need to use the correlation matrix and not the covariance one

#3. Dimensionality reduction - PCA

#to decide how may principal components to retain, we'll use the proportion of variability method
#we'll keep the number of PCs accounting for at least 95% of variability
#using prcomp as we have more variables than observations, scale is set to TRUE in order to use correlation matrix
pca.olive<-prcomp(train.set[,-1],center=TRUE,scale=TRUE)
summary(pca.olive)
#to keep 95% of variability we use three first PCs

#visualising PC1 vs. PC2 space to identify any potential outliers

plot(pca.olive$x[,1],pca.olive$x[,2], xlab="PC1", ylab="PC2", main = "Outlier detection in PC1 vs. PC2 space")

#after running the next line, click on the outlier point on the plot and press Esc
outlier<-identify(pca.olive$x[,1],pca.olive$x[,2])
outlier
#excluding the outlier from the training set
train.set2<-train.set[-outlier,]
dim(train.set2)
#rerunning PCA
pca.olive.new<-prcomp(train.set2[,-1],center=TRUE,scale=TRUE)

#checking if any outliers left

p2<-cbind(pca.olive.new$x[,1],pca.olive.new$x[,2])
p2<-as.data.frame(p2)
ggplot(p2,aes(V1,V2))+geom_point(aes(V1,V2))+labs(x="PC1",y="PC2")+
  ggtitle("PC1 vs. PC2 space after outlier removal")+theme(plot.title = element_text(hjust = 0.5))
#there are no outliers anymore, so we keep this pca model.

summary(pca.olive.new)

#the cumulative variability kept by 2 first PCs is 91.4%, it's insufficient, so we'll keep only the first 3.
#they explain 98.3% of variability

#visualising the classes in the new space
fviz_pca_ind(pca.olive.new, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = train.set2$oliveoillabels, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Regions") +
  ggtitle("2D PCA-plot from 101 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

#calculating the observations in the reduced space for the training set
scores<-pca.olive.new$x[,c(1:3)]
final.train.set<-cbind(train.set2$oliveoillabels,as.data.frame(scores))
colnames(final.train.set)[1]<-c("oliveoillabels")
str(final.train.set)

#calculating the observations in the reduced space for the validation set
final.valid.set<-predict(pca.olive.new,valid.set[,-1])
final.valid.set<-cbind(valid.set$oliveoillabels,as.data.frame(final.valid.set))
final.valid.set<-final.valid.set[,1:4]
colnames(final.valid.set)[1]<-c("oliveoillabels")
str(final.valid.set)

#calculating the observations in the reduced space for the test set
final.test.set<-predict(pca.olive.new,test.set[,-1])
final.test.set<-cbind(test.set$oliveoillabels,as.data.frame(final.test.set))
final.test.set<-final.test.set[,1:4]
colnames(final.test.set)[1]<-c("oliveoillabels")
str(final.test.set)

#4.Classification models

#4.1.k-nearest neighbours
#using 10 fold cross-validation with 3 repeats to train the model:
trctrl <- trainControl(method = "repeatedcv", number=10, repeats = 3)
knn_fit <- train(oliveoillabels ~., data = final.train.set, method = "knn",
                 trControl=trctrl, preProcess = c("center", "scale"),
                 tuneLength = 20)

knn_fit
#the best result is given by k=15, the accuracy is 64.2% based on training set

#predicting the classes, using the validation set
valid.predict<-predict(knn_fit,newdata=final.valid.set)
#assessing the predicting performance of the model
confusionMatrix(valid.predict,final.valid.set$oliveoillabels)

#Overall accuracy of the model is 64.7% with no-information rate 41%, the Kappa is 44%, so the model performs better than a simple guess, but it's still quite poor.
#It doesn't identifiy the Crete region at all, but identifies the Peloponese and Other class correctly 86% and 100% of the time respectively.
#With regards to specificity, when we shouldn't have predicted a particular class, we didn't do so correctly in 100% of cases for Crete, 
#92% for other class and 50% for Peloponese. The balanced accuracy is 50% for Crete, 96% for Other class and 68% for Peloponese region.


#4.2. Fully-grown and pruned trees

#Fully-grown tree
#creating the model
tree_grown <- rpart(oliveoillabels~PC1+PC2+PC3,
                    data = final.train.set, method = "class",
                    cp=-1,minsplit = 2, minbucket = 1)
#visualising the tree
rpart.plot(tree_grown,type=2,extra=4, tweak=1.7, under=TRUE)

#predicting the classes, using the validation set
valid.tree.predict<-predict(tree_grown,newdata=final.valid.set, type="class")
#assessing the predicting performance of the model
confusionMatrix(valid.tree.predict,final.valid.set$oliveoillabels)

#classification tree model gives us the overall accuracy of 52.9%, and the Kappa is 27%. The sensitivity is 
#20% for Crete, 80% for Other class and 57% for Peloponese region. 
#With regards to specificity, when we shouldn't have predicted a particular class, we didn't do so correctly in 83% of cases for Crete, 
#92% for Other class and 50% for Peloponese.

#let's try the pruning method:
printcp(tree_grown)
#we will prune to prune a decision tree using the cp of the largest tree that is within one
#standard deviation of the tree with the smallest xerror. 
#The smallest xerror is 0.73684, its standard deviation is 0.147697
0.73684+0.147697
#We want the largest tree with xerror less than  0.884537. There are 3 trees satisfying this rule,
#we will select the one with less splits, so the one with cp= 0.052632
tree_pruned<-prune(tree_grown,cp=.055)
#visualising the tree
rpart.plot(tree_pruned,type=2,extra=4, tweak=1.2, under=TRUE)

#predicting the classes, using the validation set
valid.tree.pruned.predict<-predict(tree_pruned,newdata=final.valid.set,type="class")

#assessing the predicting performance of the model
confusionMatrix(valid.tree.pruned.predict,final.valid.set$oliveoillabels)
#the pruned model gives a better overall accuracy at 64.7% and Kappa at 46%. The sensitivity is 40% for Crete, 100% for Other class, 
#and 57% for Peloponese region. The specificity is 92% for Crete, 83% for Other class and 70% for Peloponese.


#4.3 Random forest tree
for_tree<-randomForest(oliveoillabels~PC1+PC2+PC3,data=final.train.set,ntree=200)
for_tree

#predicting the classes, using the validation set
valid.for.tree.predict<-predict(for_tree,newdata = final.valid.set,type="class")

#assessing the predicting performance of the model
confusionMatrix(valid.for.tree.predict,final.valid.set$oliveoillabels)
#The result from random forest model seems to be similar to the fully grown tree model in terms of the overall accuracy at 52.9%, Kappa is the same at 27%.
#the sensitivity and specificity parameters are different: the model correctly identifies Crete region in 40% of cases,
#Other region in 60% of cases and Peloponese region in 57% of cases.
#With regards to specificity, the results are the following: 83% for Crete, 92% for Other class and 50% for Peloponese region.

#4.4 SVMs

#setting cost parameters
C.val<-c(0.1,0.5,1,2,5,10)

#choosing best linear SVM model
tuning.lin.svm<-tune.svm(oliveoillabels~.,data=final.train.set,type="C-classification",kernel="linear",
                         cost=C.val)
summary(tuning.lin.svm)
#the best cost parameter is 0.5, let's fit the linear SVM with this cost parameter:
lin.svm<-tuning.lin.svm$best.model

#predicting the classes, using the validation set
valid.lin.svm.predict<-predict(lin.svm,newdata=final.valid.set,type="class")

#assessing the predicting performance of the model
confusionMatrix(valid.lin.svm.predict,final.valid.set$oliveoillabels)
#The linear SVM gives us the best accuracy so far at 70.6% and Kappa - 55%. The sensitivity for Crete is 40%,
#100% for the Other class and 71% for Peloponese region. The specificity results are quite high as well:
#92% for Crete and Other class and 70% for Peloponese region.

#visualising the boundaries set by the linear SVM to predict classes:
plot(lin.svm,final.valid.set, PC1~PC2,slice=list(PC3=3), main="Linear SVM classification plot")

#radial SVM
tuning.rad.svm<-tune.svm(oliveoillabels~.,data=final.train.set,type="C-classification",kernel="radial",
                         cost=C.val,gamma = c(0.5,1,2,3,4))
tuning.rad.svm
#the best parameters are gamma 0.5 and cost 1
rad.svm<-tuning.rad.svm$best.model

#predicting the classes, using the validation set
valid.rad.svm.predict<-predict(rad.svm,newdata=final.valid.set)

#assessing the predicting performance of the model
confusionMatrix(valid.rad.svm.predict,final.valid.set$oliveoillabels)

#visualising the boundaries set by the radial SVM
plot(rad.svm,final.valid.set, PC1~PC2,slice=list(PC3=3))

#The best radial model is with cost parameter 1 and gamma 0.5. It gives us the overall accuracy of the model 
#at 64.7% and Kappa at 45%. It identifies Crete correctly in 20% of cases, Other class in 100% of cases and Peloponese region in 71% of cases.
#The specificity parameters are 100% for Crete, 83% for Other class and 60% for Peloponese region.


#Polynomial SVM
tuning.pol.svm<-tune.svm(oliveoillabels~.,data=final.train.set,type="C-classification",kernel="polynomial",
                         degree=c(2,3),coef0=c(0.5,1,2,3,4),cost=C.val,gamma = c(0.5,1,2,3))
tuning.pol.svm
summary(tuning.pol.svm)
#the best parameters for polynomial SVM are gamma 0.5, coef0 1, degree 2 and cost parameter 0.1.

pol.svm<-tuning.pol.svm$best.model

#predicting the classes, using the validation set
valid.pol.svm.predict<-predict(pol.svm,newdata=final.valid.set)

#assessing the predicting performance of the model
confusionMatrix(valid.pol.svm.predict,final.valid.set$oliveoillabels)

#visualising the polynomial SVM: 
plot(pol.svm,final.valid.set, PC1~PC2,slice=list(PC3=3))

#This model gives us the accuracy of 70.6%, which is the same that we had with the linear SVM. The Kappa is slightly lower - 54%.
#The model correctly classifies Crete region in 20% of cases, the Other class in 100% and
#Peloponese region in 86% of cases. The specificity parameters are 100% for Crete, 92% for Other class and 60% for Peloponese region.
comparison<-rbind(confusionMatrix(valid.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.tree.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.tree.pruned.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.for.tree.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.lin.svm.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.rad.svm.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")],
                  confusionMatrix(valid.pol.svm.predict,final.valid.set$oliveoillabels)$overall[c("Accuracy","Kappa")])
rownames(comparison)<-c("K-nearest neigbours", "Fully-grown tree", "Pruned tree","Random forest", "Linear SVM","Radial SVM","Polynomial SVM")
comparison<-as.data.frame(comparison)
round(comparison[order(-comparison$Accuracy,-comparison$Kappa),],3)
#The best model in terms of overall accuracy and Kappa is the linear SVM.  
#Balanced accuracy comparison
bal_acc<-rbind(confusionMatrix(valid.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.tree.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.tree.pruned.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.for.tree.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.lin.svm.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.rad.svm.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"],
                  confusionMatrix(valid.pol.svm.predict,final.valid.set$oliveoillabels)$byClass[,"Balanced Accuracy"])
rownames(bal_acc)<-c("K-nearest neigbours", "Fully-grown tree", "Pruned tree","Random forest", "Linear SVM","Radial SVM","Polynomial SVM")
bal_acc<-as.data.frame(bal_acc)
round(bal_acc[order(-bal_acc$`Class: Other`,-bal_acc$`Class: Crete`),],2)

#5.Testing of the final model on test data set
#predicting the classes, using the test set
final.test.predict<-predict(lin.svm,newdata=final.test.set)
#assessing the predicting performance of the model
confusionMatrix(final.test.predict,final.test.set$oliveoillabels)

#visualising the linear SVM classification with test set:
plot(lin.svm,final.test.set, PC1~PC2,slice=list(PC3=3))
#The model gives a better result on the test set than it gave on validation set. The overall accuracy is 86.7% and Kappa is 78%
#The model correctly classifies Crete in 50% of cases, and Peloponese and the Other class in 100% of cases.
#The model correctly doesn't indentify Crete when it's not Crete in 100% of cases, the same for the Other class. For Peloponese region
#the specificity is slightly lower - 75%. The balanced accuracy of each class is quite high as well:
#75% for Crete, 100% for Other class and 87.5% for Peloponese region.



