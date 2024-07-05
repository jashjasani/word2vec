# Word2Vec PyTorch Implementation

## Overview

This project implements Word2Vec, a fundamental technique for generating word embeddings, using PyTorch. Word2Vec converts words into fixed-size vectors, capturing semantic relationships between words in a numerical format.

## Key Concepts

### Word Embeddings
Word embeddings are dense vector representations of words that capture semantic meaning. These vectors allow us to perform mathematical operations on words, enabling various natural language processing tasks.

### Cosine Similarity
We use cosine similarity to measure the relatedness of words:
- Cosine similarity of 1: Vectors are very close (highly related words)
- Cosine similarity of 0: Vectors are unrelated
- Cosine similarity of -1: Vectors are opposite in meaning

## Approach

1. **Vocabulary Creation**: Build a vocabulary from the input text corpus.
2. **Skipgram Generation**: Create positive and negative skipgrams from the text.
3. **Sampling**: Sample skipgrams using a sampling table for efficient training.
4. **Model Training**: Train the model on generated skipgrams using CrossEntropy loss:
   - Positive pairs are labeled as 1
   - Negative skipgrams are labeled as 0

## Installation

```bash
git clone https://github.com/jashjasani/word2vec.git
cd word2vec
pip install -r ./requirements.txt
```

## Usage

1. Modify hyperparameters in `train.py` according to your needs.
2. Run the training script:

```bash
python train.py
```

## Customization

You can adjust the following hyperparameters in `train.py`:

- `embedding_size`: Dimension of word embeddings
- `window_size`: Context window size for skipgram generation
- `num_ns`: Number of negative samples
- `batch_size`: Number of samples per training batch
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for the optimizer

## Results

