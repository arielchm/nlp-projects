# nlp-projects

This is a set of Natural Language Processing projects

## Movie Review Classifier

A classification model made to predict positive or negative labels for reviews. The model was trained using the Natural Language ToolKit `movie_reviews` dataset. It uses TF-IDF Vectorization for text feature extraction and Support Vector Classification with a Linear Kernel.

## Sentiment Analysis

This project compares the predictions of the VADER (Valence Aware Dictionary for sEntiment Reasoning) model against the original labels from the `movie_reviews` dataset.

## Topic Modeling

A news articles classifier that assigns a topic from k number of topics to each article in the dataset. The Latent Dirichlet Allocation model and Non-Negative Matrix Factorization model are compared.

## Text Sequence Generator

This text generator uses a Recurrent Neural Network with Long Short-Term Memory Units trained on the first four chapters of the Moby-Dick novel to emulate it's writing style and predict a sequence based on a text seed.

## Chatbot

A chatbot console program that answers questions based on a story context using a model with an End-to-End Memory Network architecture, trained on a dataset from the bAbI project of Facebook AI Research.

```
Story > Mary is in the garden. Daniel dropped the apple in the hallway.
Question > Is the apple in the garden?
Answer >  no
```
