# Text Mining and Machine Learning on Headlines

## Overview

This project demonstrates a comprehensive approach to text mining and predictive modeling applied to a dataset of headlines. The goal is to distinguish between clickbait and non-clickbait headlines using various data science techniques, including text preprocessing, word frequency analysis, unsupervised clustering, and supervised machine learning models.

## Features

- **Text Preprocessing:** Cleansing and preparation of text data for analysis, including case normalization, punctuation removal, whitespace stripping, stemming, and removal of special characters and stopwords.
- **Word Frequency Analysis and Visualization:** Generation of word clouds to visualize the most frequent terms, and detailed frequency analysis to identify key terms.
- **TF-IDF Transformation:** Application of Term Frequency-Inverse Document Frequency (TF-IDF) to weigh the importance of terms within the corpus.
- **Unsupervised Clustering with K-Means:** Exploration of the data structure through unsupervised learning to identify natural groupings based on term characteristics.
- **Supervised Classification Models:** Training and evaluation of several machine learning models (Naive Bayes, Decision Trees, SVM) to predict whether headlines are clickbait.
- **Model Evaluation:** Use of confusion matrices to assess model performance and implementation of a majority voting system to enhance predictions.

## Installation

To run this project, you need to install the following R packages:

```R
install.packages(c('tmap', 'wordcloud', 'Rcpp', 'rgdal', 'sp', 'raster', 'stopwords', 'SnowballC', 'ClusterR', 'cluster', 'caTools', 'factoextra', 'gmodels', 'e1071', 'class', 'rpart', 'tidyverse', 'NbClust'))
```

## Usage

Load the necessary libraries:

```R
library(tm) # Text Mining
library(raster)
library(tmap)
library(dplyr)
library(ggplot2)
library(wordcloud)
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
library(tidyverse)
library(NbClust)
```

Follow the script sections to preprocess the data, perform analyses, and train models. See the script comments for detailed instructions on each step.

## Project Structure

- **Data Preprocessing:** Scripts for cleaning and preparing the text data.
- **Analysis:** Word frequency analysis, TF-IDF transformation, and unsupervised clustering scripts.
- **Model Training and Evaluation:** Scripts for training Naive Bayes, Decision Trees, SVM classifiers, and the majority voting system, including performance evaluation.
