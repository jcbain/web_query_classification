# http://blog.ideas2it.com/multi-label-classification-with-r/
# https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html
setwd('Desktop/Spring 2016/Data Mining/web_query_project/')

library(randomForest)
library(mlr)
library(rFerns)

df<-read.csv('data/queries.csv')
head(df)


x<-subset(df,select=c(char_count,verb_count,prep_count,noun_count,word_count,char_per_word,has_num,sup1))

rf<-randomForest(sup1~.,data=x,importance=FALSE,proximity=FALSE)

yeast = getTaskData(yeast.task)
labels = colnames(yeast)[1:14]
yeast.task = makeMultilabelTask(id = "multi", data = yeast, target = labels)
yeast.task
multilabel.lrn = makeLearner("classif.rpart", predict.type = "prob")
multilabel.lrn = makeMultilabelBinaryRelevanceWrapper(multilabel.lrn)
multilabel.lrn
multilabel.lrn1 = makeMultilabelBinaryRelevanceWrapper("classif.rpart")
multilabel.lrn1
multilabel.lrn2 = makeLearner("multilabel.rFerns")
multilabel.lrn2
mod = train(multilabel.lrn, yeast.task)
mod = train(multilabel.lrn, yeast.task, subset = 1:1500, weights = rep(1/1500, 1500))
mod
mod2 = train(multilabel.lrn2, yeast.task, subset = 1:100)
mod2
pred = predict(mod, task = yeast.task, subset = 1:10)
pred = predict(mod, newdata = yeast[1501:1600,])
pred2 = predict(mod2, task = yeast.task)
names(as.data.frame(pred2))
performance(pred)
performance(pred2, measures = list(hamloss, timepredict))
