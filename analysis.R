# http://blog.ideas2it.com/multi-label-classification-with-r/
# https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html
setwd('Desktop/Spring 2016/Data Mining/web_query_project/')

library(randomForest)
library(mlr)
library(rFerns)
library(ROCR)
library(nnet)
library(party)
library(extraTrees)
library(LiblineaR)
library(e1071)

#library(deepnet)
#library(FSelector)

##########################################################################
# read in the data frame and perform some subsets to easy classification # 
##########################################################################
df<-read.csv('data/queries.csv')

#####################################################################
# find the information gain of each classification label separately #
#####################################################################
labs<-subset(df,select=c(char_count,prep_count,verb_count,meaning_count,word_count,char_per_word,has_num,label6))
labels = colnames(labs)[9]
task= makeClassifTask(id = "classify", data = labs, target = labels)
fv = generateFilterValuesData(task, method = "information.gain")
fv

# full subset
x<-subset(df,select=c(char_count,prep_count,noun_count,verb_count,word_count,char_per_word,meaning_count,has_num,label1,label2,label3,label4,label5,label6,label7))

# partial subset
x<-subset(df,select=c(char_count,prep_count,has_num,char_per_word,label1,label2,label3,label4,label5,label6,label7))

# max information gain
#     name        type        information.gain
#1    char_count  integer     0.004742764 (lab7)
#2    prep_count  integer     0.004949343 (lab7)
#3    noun_count  integer     0.000000000
#4    verb_count  integer     0.000000000
#5 meaning_count  integer     0.09570352 (lab6)
#6    word_count  integer     0.000000000
#6 char_per_word  numeric     0.007672985 (lab2)

# decision tree
query = x
labels = colnames(query)[5:11]
query.task = makeMultilabelTask(id = "multi", data = query, target = labels)
query.task
multilabel.lrn = makeLearner("classif.rpart", predict.type = "prob")
multilabel.lrn = makeMultilabelBinaryRelevanceWrapper(multilabel.lrn)
multilabel.lrn
multilabel.lrn1 = makeMultilabelBinaryRelevanceWrapper("classif.rpart")
multilabel.lrn1
multilabel.lrn2 = makeLearner("multilabel.rFerns")
multilabel.lrn2
mod = train(multilabel.lrn, query.task)
mod = train(multilabel.lrn, query.task, subset = 1:2000, weights = rep(1/2000, 2000))
mod
mod2 = train(multilabel.lrn2, query.task, subset = 1:2000)
mod2
pred = predict(mod, task = query.task, subset = 1:10)
pred = predict(mod, newdata = x[2001:2400,])
pred2 = predict(mod2, task = query.task)
names(as.data.frame(pred2))
performance(pred)
performance(pred2, measures = list(hamloss, timepredict))
getMultilabelBinaryPerformances(pred, measures = list(acc, mmce, auc,fn,fp,tp,tn,ppv,tpr,f1))

# neural network
multilabel.lrn3 = makeLearner("classif.avNNet", predict.type = "prob")
multilabel.lrn3 = makeMultilabelBinaryRelevanceWrapper(multilabel.lrn3)
mod3 = train(multilabel.lrn3, query.task)
pred = predict(mod3, newdata = x[2001:2400,])
performance(pred)
getMultilabelBinaryPerformances(pred, measures = list(acc, mmce, auc,fn,fp,tp,tn,ppv,tpr,ppv,tpr,f1))

# 5 fold CV neural network
rdesc = makeResampleDesc(method = "CV", stratify = FALSE, iters = 5)
r = resample(learner = multilabel.lrn3, task = query.task, resampling = rdesc, show.info = FALSE)
r
names(r)
getMultilabelBinaryPerformances(r$pred,measures = list(acc, mmce, auc,fn,fp,tp,tn,ppv,tpr,f1))
j<-as.data.frame(getMultilabelBinaryPerformances(r$pred,measures = list(acc, mmce, auc,fn,fp,tp,tn,ppv,tpr,f1)))
write.csv(j,'output_nn.csv')
