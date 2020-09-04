library(GGally)
library(plotly)
library(plyr)
library(GGally)
library(waffle)
library(ModelMetrics)
library(ROCR)
library(randomForest)
library(dplyr)
#install.packages("ipred")
library(ipred)
library(caret)
library(gbm)
#install.packages("heuristica")
library(heuristica)

#Load Datasets
math<-read.csv("A:/OR/568/Term project/student-mat.csv")
portu<-read.csv("A:/OR/568/Term project/student-por.csv")

#combine datasets
students<-rbind(math,portu)

#summary
summary(students)

#View
View(students)

#Data Preprocessing
students$G3int<-cut(students$G3,c(-1,10,20))
students$Gavg<-(students$G1+students$G2+students$G3)/3
summary(students)                
str(students)

#correlation analysis
corp<-(cor(students[sapply(students, function(x) !is.factor(x))]))
corrplot(corp,method="number",order="hclust")

ggcorr(corp, method = c("everything", "pearson"))+  ggtitle("Correlation Analysis")


#Data Visualizations
#Daily and weekly alcohol consumption vs final grades
plot_ly(data=students, x = students$Dalc,y=students$G3, type = "box", name="Dalc")%>%
  add_trace(data=students, x =students$Walc, y =students$G3 ,type = "box", name="Walc")%>%
  layout(boxmode="group",xaxis=list(title="Alcohol level"),yaxis=list(title="G3"))

#Factorization
students$Dalc<-as.factor(students$Dalc)
students$Dalc<-mapvalues(students$Dalc, from =1:5, to=c("Very Low","Low","Medium","High","Very High"))

students$Walc<-as.factor(students$Walc)
students$Walc<-mapvalues(students$Walc, from =1:5, to=c("Very Low","Low","Medium","High","Very High"))

#assigning to fill colors
waffle.col <- c("#00d27f","#adff00","#f9d62e","#fc913a","#ff4e50")

#workday alochol consumptions and grades
p1<-ggplot(students, aes(x=Dalc, y=G1, fill=Dalc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("First period grade")

p2<-ggplot(students, aes(x=Dalc, y=G2, fill=Dalc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Second period grade")

p3<-ggplot(students, aes(x=Dalc, y=G3, fill=Dalc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Final period grade")

grid.arrange(p1,p2,p3,ncol=3)

#weekend alcohol consumption and grades
p4<-ggplot(students, aes(x=Walc, y=G1, fill=Walc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("First period grade")

p5<-ggplot(students, aes(x=Walc, y=G2, fill=Walc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Second period grade")

p6<-ggplot(students, aes(x=Walc, y=G3, fill=Walc))+
  geom_boxplot()+
  theme_bw()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Final period grade")

grid.arrange(p4,p5,p6,ncol=3)

#daily and weekly alcohol consumption of gender vs final grade
ggplot(students,aes(x=sex,y=G3,fill=students$Dalc))+geom_bar(stat="identity")+xlab("Sex")+ylab("Final Grades")+ggtitle("Daily alcohol consumption of Male and Female vs Final Grades")
ggplot(students,aes(x=sex,y=G3,fill=students$Walc))+geom_bar(stat="identity")+xlab("Sex")+ylab("Final Grades")+ggtitle("Weekly alcohol consumption of Male and Female vs Final Grades")


#PCA analysis

pca<-subset(students,select=c(age,Medu,Fedu,traveltime,studytime,failures,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3))
pc.out<-prcomp(pca,scale=TRUE)            
pc.out
pc<-names(pc.out)
pc
pc.out$center
pc.out$scale
pc.out$rotation=-pc.out$rotation
pc.out$rotation

biplot(pc.out)

pc.out$sdev
pr.var = pc.out$sdev^2
pr.var
pve = pr.var/sum(pr.var)
pve
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained",
     ylim=c(0,1), type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained",
     ylim=c(0,1), type='b')
pve.df<-as.data.frame(pve)

plot_ly(x=cumsum(pve),type="scatter", name="Cumulative PVE", mode="lines+markers", 
        marker = list(size = 6,color = 'red'), 
        line = list(color = '')
) %>%
  layout(yaxis = list(title="Number of principal components"))


#CLASSIFICATION MODELS

#LOGISTIC REGRESSION

stud<-subset(students,select=c(age,Medu,Fedu,traveltime,studytime,failures,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3,G3int))
set.seed(12345)
row<-nrow(stud)
trainindex<-sample(row,0.60*row,replace = FALSE)
training<-stud[trainindex,]
validation<-stud[-trainindex,]
logistic<-glm(G3int~age+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,family=binomial)
summary(logistic)


mylogit.step = step(logistic, direction='backward')
bestfit<-glm(G3int ~ studytime + failures + Walc + G1 + G2,data=training,family=binomial)
summary(bestfit)

#best fit accuracy 
mylogit.probs1<-predict(bestfit,validation,type="response")
mylogit.pred2 = rep("BELOW AVERAGE", 0.4*row)
mylogit.pred2[mylogit.probs1 >0.5] = "good grades"
table(mylogit.pred2, validation$G3int)

#accuracy considering all important variables
mylogit.probs<-predict(logistic,validation,type="response")
mylogit.probs
mylogit.pred = rep("BELOW AVERAGE", 0.4*row)
mylogit.pred[mylogit.probs >0.5] = "good grades"
table(mylogit.pred, validation$G3int)

#ROC curve
mydf <-cbind(validation,mylogit.probs)
logit_scores <- prediction(predictions=mydf$mylogit.probs, labels=mydf$G3int)
logistic_perf<-performance(logit_scores,"tpr","fpr")
plot(logistic_perf,
     main="ROC Curves",
     xlab="1 - Specificity: False Positive Rate",
     ylab="Sensitivity: True Positive Rate",
     col="red",  lwd = 10)
abline(0,1, lty = 300, col = "blue",  lwd = 5)
grid(col="black")
logistic_auc <- performance(logit_scores, "auc")
logistic_auc


#RANDOM FOREST

fitting2<-randomForest(G3int~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,ntree=500,type="class",mtry=3)
plot2<-varImpPlot(fitting2,type=2)
plot2
predictrand<-predict(fitting2,validation,type="class")
cm<-confusionMatrix(validation$G3int,predictrand)
cm1<-table(validation$G3int,predictrand)
cm1
cm

#removing important variables g1,g2
fit12<-randomForest(G3int~.-G2-G1-G3,data=training,ntree=500,type="class",mtry=3)
plot2<-varImpPlot(fit12,type=2)
pre<-predict(fit12,validation,type="class")
cm12<-table(validation$G3int,pre)
cm12
predictrand[1:5]

#BAGGING

BaggTree= bagging(G3int~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training)
Bagg_yHat = predict(BaggTree,validation)
BaggPR = postResample(pred=Bagg_yHat, obs=validation$G3int)
BaggPR
cmb= confusionMatrix(validation$G3int,Bagg_yHat)
cmb

#DECISION TREE
library(rpart)
library(rpart.plot)
fitting12<-rpart(G3int~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,method="class")
plot<-rpart.plot(fitting12,type=2,extra="auto")
pred<-predict(fitting12,validation,type="class")
cm2<-confusionMatrix(validation$G3int,pred)
cm2
as<-table(validation$G3int,pred)
as


#GRADIENT BOOSTING

gradiant<-train(G3int~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,method='gbm',trControl=trainControl(method="cv",number=10),verbose=FALSE)
gradiant

summary(gradiant)
predicgrad<-predict(gradiant,newdata=validation)
gradpred<-table(predicgrad,validation$G3int)
gradpred
predicgrad[1:5]



