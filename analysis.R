# http://blog.ideas2it.com/multi-label-classification-with-r/
setwd('Desktop/Spring 2016/Data Mining/web_query_project/')

library(randomForest)
library(RWeka)

df<-read.csv('data/queries.csv')
head(df)


x<-subset(df,select=c(char_count,verb_count,prep_count,noun_count,word_count,char_per_word,has_num,sup1))

rf<-randomForest(sup1~.,data=x,importance=FALSE,proximity=FALSE)

