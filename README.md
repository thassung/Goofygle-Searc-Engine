# Search Engine Demo App

## Overview

Welcome to the Search Engine Demo App! This web-based application demonstrates the basic functionality of a search engine, allowing users to input a query and retrieve the top 10 most similar passages/word from trained corpus.

## Features

- **Input:** Users can enter search queries in the provided search bar.
- **Search Engine:** Users can select pretrained LM to use for searching including Skipgram, Skipgram with Negative Sampling, GloVe, and Glove (Gensim)
- **Submit Button:** Users click submit after typing the input and select the search engine. The model will search for the most similar words
- **Search Result:** The 10 most similar passages are returned as a text next to the submit button.
- **Dataset:** The language models were trained with data from nltk.corpus.reuters.sents(). 

### Prerequisites

- Ensure you have python, flask, torch, gensim, numpy, and sklearn installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thassung/Goofygle-Search-Engine.git
   ```

2. Install the required Python dependencies:

   ```bash
   pip install flask torch gensim numpy scikit-learn
   ```

3. Navigate to the app directoty:
   ```bash
   cd Goofygle-Search-Engine
   cd app
   ```

4. Start the flask application:
   ```bash
   python main.py
   ```

### LM performance

__Analogy Test__
| Model             | Window Size | Training Loss | Training Time (sec) | Semantic Accuracy | Syntactic Accuracy |
|-------------------|--------------|---------------|----------------------|-------------------|--------------------|
| Skipgram          | 2            | 13.729        | 1084.5               | 0                 | 0                  |
| Negative Sampling | 2            | 12.891        | 1105.4               | 0                 | 0                  |
| GloVe             | 2            | 19.204        | 47                   | 0                 | 0                  |
| GloVe (Gensim)    | -            | -             | -                    | 0.9387            | 0.5545             |

__Similarity Test__
| Model          | MSE   | Spearman Rank Corr.  |
|----------------|-------|----------------------|
| Skipgram       | 18.725| -0.074 (p = 0.426)   |
| NEG            | 16.472| -0.038 (p = 0.681)   |
| GloVe          | 20.375| -0.012 (p = 0.945)   |
| GloVe (gensim) | 8.367 | 0.601 (p = 1.3e-20)  |
| Y_true         | 0     | 1                    |

One thing to note here is that the corpus size used in model training are not the same. The Skipgram and Negative Sampling models are train with data from nltk.corpus.reuters. The GloVe model is trained with the same dataset but only the first 1,000 sentences from the avaiable 54,716 sentences in the corpus due to the shortage of memory required to initialize the co-occurance matrix. GloVe (Gensim) model is pretrained and can be obtain from [here](https://nlp.stanford.edu/projects/glove/).

Semantic and syntactic tests are performed using data from [here](https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt). Semantic tests are performed using data in **capital-common-countries** section and syntaction tests are performed using data in **gram7-past-tense** section. The accuracy is calculated using the correct inferences and the total inferences where every word in the test is known by the LM.

