# End-to-End Algorithmic Stock Prediction Pipeline

An advanced, production-oriented quantitative framework designed for multi-asset price forecasting and algorithmic alpha generation. This repository integrates a custom Transformer-based deep learning architecture with automated raw alternative data harvesting (NLP financial news sentiment analysis) and real-time market data orchestration.

## 🏛️ System Architecture & Repository Layout

The system is structurally split into decoupled data ingestion, tokenization, text classification, and sequential modeling modules:

*   **`/transformer`**: Core deep learning engine featuring a custom implementation of sequence-to-sequence Transformer layers optimized for high-frequency financial time-series forecasting.
*   **`/news_crawler`**: Distributed data harvesting module built to systematically crawl and parse unstructured alternative textual datasets (financial news portals, corporate press releases).
*   **`/news_classifier`**: NLP classification pipeline that ingests crawled textual streams and scores latent market sentiment to serve as dynamic feature weights.
*   **`market_data.py`**: Robust data engineering interface for extracting, cleaning, adjusting, and aligning raw historical and spot financial price vectors.
*   **`learn_bpe.py` & `apply_bpe.py`**: Byte Pair Encoding (BPE) subword tokenization infrastructure optimized to preprocess raw text corpora for efficient neural vocabulary encoding without vocabulary explosion.

## 🛠️ Key Technical Features

### 1. Transformer-Driven Time-Series Modeling
Unlike standard text-based language models, the sequence-to-sequence architecture inside `/transformer` is optimized to process multi-dimensional inputs. It ingests historical price features (OHLCV, volatility, funding costs) concatenated with our engineered NLP sentiment metrics to predict future price trajectories and capture complex, non-linear dependencies across lookback windows.

### 2. Alternative Data & Sentiment NLP Pipeline
*   **Subword Tokenization:** BPE tokenization (`learn_bpe.py`) trains a compressed vocabulary on financial domain text to cleanly handle complex market terminology and acronyms.
*   **Latent Sentiment Attribution:** The classification layer evaluates incoming raw news signals on the fly, outputting structured alpha indicators (bullish/bearish distributions) that function as volatility modulators in the primary pricing engine.

### 3. Quantitative Risk & Data Engineering
The framework utilizes strict data transformations via `market_data.py` to prevent data leakage—a critical pitfall in financial ML. All metrics are strictly forward-aligned and normalized using rolling z-score techniques to maintain stationarity across varying market regimes.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   PyTorch / TensorFlow 2.x
*   NumPy / Pandas

### Installation & Execution Blueprint
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd Stock_prediction
   ```

2. Run the tokenization and text classification pipeline to extract sentiment indices:
   ```bash
   python learn_bpe.py --input raw_corpus.txt --output vocab.bpe
   python apply_bpe.py --vocab vocab.bpe --text input_news.txt
   ```

3. Initialize market data structures and trigger the sequential Transformer pipeline:
   ```bash
   python market_data.py --ticker MSFT --start 2021-01-01
   python transformer/train.py --config config.yaml
   ```

## ⚖️ Core Philosophy

This architecture operates on strict determinism and high performance. The code shifts away from standard black-box wrappers toward exposed tensor manipulation, making it highly customizable for exotic derivatives modeling, high-frequency limit order book (LOB) dynamics, and dynamic portfolio allocation constraints.
