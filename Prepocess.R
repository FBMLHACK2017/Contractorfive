data<-read.csv('/Users/jianjiey/Documents/Building_Permits___Current.csv',header=T)
all.list<-NULL
word.list<-list(NULL)
l<-dim(data)[1]
for(i in 1:l){
  temp<-strsplit(as.character(data$Description[i]),' ')
  l.temp<-length(temp[[1]])
  a<-NULL
  for(j in 1:l.temp){
    a.new<-wordStem(tolower(gsub('[[:punct:]]', '',temp[[1]][j])))
    a<-c(a,a.new)
  }
  word.list[[i]]<-a
  all.list<-c(all.list,a)
}

uni.list<-unique(all.list)  

for(i in 1:l){
  which$word.list[[i]]
}

x <- sort.int(table(all.list),decreasing=T)
x<-as.data.frame(x)[1:5000,]
uni.list<-as.character(x[,1])


ct.table<-matrix(rep(0,l*5000),c(l,5000))
#names(ct.table)=uni.list
#paste('ct.table$',word.list[[i]][j])
for(i in 1:l){
  for(j in 1:length(word.list[[i]])){
    idx<-which(uni.list==word.list[[i]][j])
    ct.table[i,idx]=ct.table[i,idx]+1
  }
}


install.packages('SnowballC')
library(SnowballC)


write.csv(ct.table,'countTB.csv')
write.csv(pTag,'pTag.csv')
write.csv(uni.list,'keywords.csv')

tolower()
wordStem('have')
pTag<-rep(0,l)
for(i in 1:l){  
  pTag[i]<-as.numeric(gsub("[[:punct:]]",'', data$Value[i]))
}

keywords

