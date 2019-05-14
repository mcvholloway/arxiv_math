# Applying NLP to the arXiv Mathematics Preprint Repository
This repository contains code for a capstone project done for Cohort 2 of the Data Science Bootcamp at Nashville Software School.

arXiv.org is an online repository for preprints of research papers in mathematics, physics, computer science, and other fields. In this project, metadata from nearly 400,000 papers from the field of mathematics was obtained using the arXiv API. 

This project had three main objectives:

1. Build a classification model to predicting the categories to which a paper belongs.
2. Create a recommendation system to find papers related to and/or similar to a given paper.
3. Develop an unsupervised keyword/keyphrase extractor to search for papers by topic.

### Objective 1: Classification Model
The first goal was to predict the category of a paper using only its title and abstract. Adding to the challenge, about 30% of papers fall into more than one category. For example, a paper might be considered both Algebraic Geometry and Representation Theory.
A number of models were tried, including LinearSVC, Logistic Regression, Word- and Character-Level CNNs, and Facebook's FastText vectors, but for the final implementation, a One-vs-Rest Logistic Regression model was chosen.

To address the class imbalance, a random upsampler designed for multilabel datasets and described in [_Addressing imbalance in multilabel classification: Measures and random resampling algorithms_ by Charte, et. al.](https://www.sciencedirect.com/science/article/pii/S0925231215004269) was implemented. The idea is to measure how out of balance each class is and to upsample to bring those classes most out of balance up to the average imbalance level.

Finally, to address the multilabel aspect of the dataset, a classifier chain algorithm was tried, as described in [_Classifier Chains for Multi-label Classification_ by Read, et. al.](https://www.cs.waikato.ac.nz/~eibe/pubs/chains.pdf) was tried. This resulted in a slightly higher macro F1 score, but was ultimatly not used in the final implementation since the sequential nature of the model made it slower than a one-vs-rest classifier.

### Objective 2: Recommendation System
The goal of the recommender is to locate papers that may be related to a given paper. This is accomplished by finding the most similar papers to a given one based on the cosine similarities of term-frequency inverse-document frequency vectors. I also included code that allows the user to search based on topics or keywords, while still ranking the search results based on similariy to a given paper.

### Objective 3: Unsupervised Tagger
For purposes of classification and searching, it is useful to generate tags or keywords/phrases for each paper. I combined ideas from [_Key2Vec: Automatic Ranked Keyphrase Extraction from Scientific Articles using Phrase Embeddings_](https://aclweb.org/anthology/N18-2100) and [_YAKE! Collection-Independent Automatic Keyword Extractor_](http://yake.inesctec.pt/) to extract keywords and phrases from the title and abstract.

### Implementation
The tools developed above were combined and implemented in an app built using Flask.





