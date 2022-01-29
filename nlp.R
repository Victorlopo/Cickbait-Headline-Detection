# Install packages

install.packages('tmap')
install.packages('wordcloud') # WordCloud Study
install.packages(c("Rcpp", "rgdal", "sp", "raster"))
install.packages('stopwords') # Stopwords
install.packages("SnowballC") # WordCloud Study
install.packages("ClusterR")
install.packages("cluster")
install.packages("caTools")
install.packages("factoextra")
install.packages('gmodels')
install.packages("e1071") # SVM classifier
install.packages("class")
install.packages("rpart")
install.packages('tidyverse') # Silhouette Analysis
install.packages('NbClust') # NbClustering Analysis



# Load packages

library(tm) # Text Mining functionalities
library(raster)
library(tmap) # For the corpus cleaning
library(dplyr)
library(ggplot2)
library(wordcloud) # Wordcloud Study
library(caTools)
library(factoextra)
library(gmodels)
library(e1071) # SVM classifier
library(class)
library(reshape2)
library(readr)
library(stopwords)
library(wordcloud)
library(SnowballC)
library(ClusterR)
library(cluster)
library(rpart)
library(tidyverse) #Silhouette analysis
library(NbClust) #NbClustering analysis


# Functions used

# Function 1 for removing special characters

removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]","",x) #remove everything that is not alphanumerical symbol or space

# Function 2 for creating the word frequency plot

Word_freq<-function(TDM){
  docmatrix <- as.matrix(TDM)
  doc.counts <- rowSums(docmatrix)
  doc.df <- data.frame(cbind(names(doc.counts),as.numeric(doc.counts)),stringsAsFactors = FALSE)
  names(doc.df) <- c("Terms", "Frequency")
  doc.df$Frequency <- as.numeric(doc.df$Frequency)
  doc.occurrence <- sapply(1:nrow(docmatrix),
                           function(i)
                           {
                             length(which(docmatrix[i, ] > 0)) / ncol(docmatrix)
                           })
  doc.density <- doc.df$Frequency / sum(doc.df$Frequency)
  
  # Add the term density and occurrence rate
  doc.df <- transform(doc.df,density = doc.density,ocurrencia =doc.occurrence)
  S=head(doc.df[with(doc.df, order(-Frequency)),], n=30)
  return(S)
}

# Load Data

raw_data <- read.csv('FinalDataset.csv', encoding='UTF-8') # Read csv with the data
raw_data <- raw_data[ , !(names(raw_data) %in% c('X'))] # Eliminate a useless column ('X')
head(raw_data)
summary(raw_data)
table(raw_data$clickbait) # See if the dataset is balanced (3633 no clickbait and 3367 yes clickbait)

corpus = Corpus(VectorSource(raw_data$headline)) # Convert the raw_data df into a corpus (just with the headlines)
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower) # Convert headlines to lowercase
corpus = tm_map(corpus, removePunctuation) # Remove punctuation marks from a text document
corpus = tm_map(corpus, stripWhitespace) # Strip extra whitespace from a text document
corpus = tm_map(corpus, stemDocument) # Stem words in a text document using Porter's stemming algorithm
corpus <- tm_map(corpus, removeSpecialChars) # Use the function 1 to eliminate " (for example: Time", 2015", "new ...)

# Wordcloud Study

TermDocMat = TermDocumentMatrix(corpus) 
TermDocMat_matrix <- as.matrix(TermDocMat) 
words <- sort(rowSums(TermDocMat_matrix),decreasing=TRUE) # Order the words to do the Wordcloud
df_wordcloud <- data.frame(word = names(words),freq=words)
wordcloud(words = df_wordcloud$word, freq = df_wordcloud$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

# Remove some irrelevant but with high frequency words seen in the wordcloud
corpus <- tm_map(corpus, removeWords, c(stopwords("english"),"The", "For", "That", "Will", "And", "This","Which", "Make","With", "About"))
TermDocMat = TermDocumentMatrix(corpus) 
TermDocMat_matrix <- as.matrix(TermDocMat) 
words <- sort(rowSums(TermDocMat_matrix),decreasing=TRUE) # Order the words to do the Wordcloud
df_wordcloud <- data.frame(word = names(words),freq=words)
wordcloud(words = df_wordcloud$word, freq = df_wordcloud$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

# Term Frequency study

Freq_plot=Word_freq(TermDocMat) # Uses Function 2 (defined on top of the document)
ggplot(Freq_plot,aes(Freq_plot$Frequency,factor(Freq_plot$Terms,levels=Freq_plot$Terms)))+geom_point(stat="identity", colour="red")+ggtitle("Frequency Table")+xlab("Frequency of the word")+ylab("30 more frequent words")

# Creation of the tfidf matrix and its conversion to a DataFrame

tdm.tfidf = TermDocumentMatrix(corpus, control = list(weighting = weightTfIdf)) # TF-IDF matrix
set.seed(123)
sparse = removeSparseTerms(tdm.tfidf, 0.995) # Reduce the matrix (so many sparse terms)
tSparse = as.data.frame(as.matrix(sparse))
tSparse <- data.frame(t(tSparse)) # Creation of the dataFrame to train the models
colnames(tSparse) = make.names(colnames(tSparse))


# K-MEANS UNSUPERVISED UNDERSTANDING OF DATA


# Elbow Method 

set.seed(123)
fviz_nbclust(tSparse, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")


# Average Silhouette Method

set.seed(123)
fviz_nbclust(tSparse, kmeans, method = "silhouette")


# Gap Method

set.seed(123)
res1<-NbClust(tSparse, distance = "euclidean", min.nc=2, max.nc=7, method = "kmeans", index = "gap")  
NbCluster_result = res1$Best.nc[1] # Returns the number of estimated clusters (2)
print(NbCluster_result)

# K MEANS

k2 <- kmeans(tSparse, centers = 2, nstart = 25) # Creation of the k-means model with 2 clusters
fviz_cluster(k2, data = tSparse) # Visualize the clusters formed



# SUPERVISED LEARNING

# Preparation of data for the supervised tasks

tSparse$recommended_id = raw_data$clickbait # Add the labels to the items
prop.table(table(tSparse$recommended_id)) # See the probability of each class
set.seed(123)
split = sample.split(tSparse$recommended_id, SplitRatio = 0.7) # 70 % of the dataset in the train and 30 % in the test
trainSparseWithLabels = subset(tSparse, split==TRUE) # Train set with labels
testSparseWithLabels = subset(tSparse, split==FALSE) # Test set with labels
ytrainSparse = trainSparseWithLabels$recommended_id # Labels of the train set
ytestSparse = testSparseWithLabels$recommended_id # Labels of the test set
trainSparseNoLabels <- trainSparseWithLabels[ , !(names(trainSparseWithLabels) %in% c('recommended_id'))] # Train set without labels
testSparseNoLabels <- testSparseWithLabels[ , !(names(testSparseWithLabels) %in% c('recommended_id'))] # Test set without labels

# Naive Bayes classifier

naive_classifier <- naiveBayes(trainSparseNoLabels, ytrainSparse) # Creation of the Naive Bayes model and training
data_test_pred_NB <- predict(naive_classifier, testSparseNoLabels) # Predict labels on the test set
table(data_test_pred_NB, ytestSparse,dnn=c("Prediction","Actual")) # Table with the confusion matrix

# Decision Tree classifier

tree_classifier <- rpart(recommended_id~., data = trainSparseWithLabels, method = 'class') # Creation of the Decision Tree model and training
data_test_pred_DT <- predict(tree_classifier, testSparseNoLabels, type = 'class') # Predict labels on the test set
table(data_test_pred_DT, ytestSparse,dnn=c("Prediction","Actual")) # Table with the confusion matrix

# SVM Classifier

svm_classifier = svm(formula = recommended_id ~ .,
                 data = trainSparseWithLabels,
                 type = 'C-classification',
                 kernel = 'linear') # Creation of the SVM model and training
data_test_pred_SVM = predict(svm_classifier, newdata = testSparseNoLabels) # Predict labels on the test set
table(data_test_pred_SVM, ytestSparse,dnn=c("Prediction","Actual")) # Table with the confusion matrix

# Majority Voting

data_test_pred_voting<-as.factor(ifelse(data_test_pred_NB=='1' & data_test_pred_DT=='1','1',ifelse(data_test_pred_NB=='1' & data_test_pred_SVM=='1','1',ifelse(data_test_pred_DT=='1' & data_test_pred_SVM=='1','1','0')))) # Hard voting
table(data_test_pred_voting, ytestSparse,dnn=c("Prediction","Actual")) # Table with the confusion matrix
