### Stock Prediction with Financial News Sentiment

This repository implements a high-performance text-mining pipeline for asset return prediction, comparing the econometric SESTM algorithm with modern Transformer-based architectures. The foundational framework is adapted from the National Bureau of Economic Research working paper Predicting Returns With Text Data by Zheng Tracy Ke, Bryan T. Kelly, and Dacheng Xiu. 

### Project Overview

Quantifying unstructured textual information remains a central challenge in empirical finance. This project provides a comparative implementation of two distinct methodologies to extract return-predictive sentiment signals from financial news text: 

1. **SESTM (Sentiment Extraction via Screening and Topic Modeling):** A transparent, supervised learning methodology combining correlation screening, two-topic modeling, and penalized likelihood estimation.
2. **Transformer-Based Sentiment Models:** An extension designed to capture contextual word interactions, semantic nuances, and long-range dependencies that traditional bag-of-words models omit.

### Core Methodology: SESTM

The SESTM algorithm bypasses generic, ad hoc financial dictionaries in favor of a context-specific, supervised scoring pipeline consisting of three steps: 

### 1. Feature Screening

Isolates sentiment-charged words from a high-dimensional vocabulary. It calculates the marginal frequency with which a specific word co-occurs with positive daily stock returns. Words exceeding or falling below optimized significance thresholds are retained, discarding sentiment-neutral vocabulary noise. 

### 2. Supervised Topic Modeling

Assigns term-specific weights by fitting a generative, likelihood-based two-topic mixture model to the screened word counts. It extracts a frequency vector representing overall word usage and a tone vector representing the relative positive or negative orientation of each word. 

### 3. Article Scoring

Aggregates term frequencies into a singular, document-level sentiment score using a penalized maximum likelihood estimator. The implementation uses a Beta-distributed prior to shrink scores toward neutrality when articles contain sparse sentiment-charged text, stabilizing out-of-sample portfolio predictions. 

### Transformer Enhancement

While SESTM is highly scalable and interpretable, its underlying bag-of-words assumption ignores word order and context. This repository introduces deep learning alternatives to improve predictive boundaries: 

* **Contextual Embeddings:** Captures shifts in term meaning based on surrounding sentence structure, resolving misclassifications caused by negation or domain-specific idioms.
* **Attention Mechanisms:** Dynamically determines the relative importance of phrases across entire articles, mimicking how market participants prioritize core news elements over auxiliary data.
* **Supervised Return Fine-Tuning:** Pre-trained sequence classification models are fine-tuned using asset return signs as direct training targets, preserving the supervised essence of the original paper.

### Getting Started

### Prerequisites

* Python 3.8 or higher
* PyTorch
* Hugging Face Transformers
* NLTK
* Scikit-learn
* Pandas and NumPy

### Execution Pipeline

1. Run the data preprocessing script to normalize, stem, and tokenize raw news feeds.
2. Execute the SESTM script to perform marginal correlation screening and maximum likelihood optimization.
3. Execute the Transformer notebook to fine-tune text classification heads on historical return benchmarks.
4. Run the backtesting script to simulate an open-to-open daily long-short trading strategy based on generated signals.
