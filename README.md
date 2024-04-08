# Information Retrieval System (Python)

## Overview
This project is an Information Retrieval System implemented in Python. It reads text content from .txt files, tokenizes it using the NLTK library, constructs an inverted index, allows users to search the files, and displays search results based on cosine similarity.

## NLTK (Natural Language Toolkit)
NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet. Additionally, NLTK includes a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, making it a powerful tool for natural language processing tasks.

## Inverted Index
An inverted index is a data structure used to map content, such as words or numbers, to their locations in a database file or a set of documents. In the context of information retrieval systems, an inverted index is constructed by indexing the terms found in the documents and storing references to the documents where each term appears. This allows for efficient search operations, as documents containing specific terms can be quickly identified.

## Cosine Similarity
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. In the context of information retrieval, cosine similarity is often used to determine the similarity between documents or queries based on their vector representations in a high-dimensional space. It is a popular technique for ranking search results, as it effectively captures the semantic similarity between documents or queries.

## How to Run
1. Ensure you have Python installed on your system.
2. Install the necessary dependencies (NLTK, natsort, numpy)
3. Run the main Python script to start the program.
