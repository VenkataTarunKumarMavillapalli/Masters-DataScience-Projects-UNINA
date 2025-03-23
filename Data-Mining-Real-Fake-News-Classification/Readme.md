# Data Mining Final - Fake News and Real News Classification

## Overview
This repository contains the classification of fake and real news articles using various machine learning and deep learning models. The study compares two text representations—**TF-IDF** and **Word2Vec**—to evaluate their effectiveness across different classifiers. The results demonstrate that TF-IDF generally outperforms Word2Vec in this task.

## Table of Contents
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Analysis and Key Findings](#analysis-and-key-findings)
- [Conclusion and Recommendations](#conclusion-and-recommendations)

## Methodology

### Text Representations
- **TF-IDF**: Captures term frequency and document importance, making it effective for classification tasks.
- **Word2Vec**: Generates dense vector representations of words based on semantic relationships.

### Models Evaluated
- **Traditional Machine Learning Models**: Logistic Regression, Random Forest
- **Neural Network Models**: Multi-Layer Perceptron (MLP), Recurrent Neural Networks (RNN, LSTM, GRU), Bidirectional Variants, Transformer-based models

## Experimental Results

### Traditional Machine Learning Models
| Model | Representation | Accuracy |
|--------|---------------|-----------|
| Logistic Regression | TF-IDF | **93.03%** |
| Logistic Regression | Word2Vec | 90.84% |
| Random Forest | TF-IDF | **91.81%** |
| Random Forest | Word2Vec | 90.67% |

### Neural Network Models
| Model | Representation | Accuracy |
|--------|---------------|-----------|
| MLP | TF-IDF | **94.16%** |
| MLP | Word2Vec | 90.51% |
| RNN | TF-IDF | 92.00% |
| LSTM | TF-IDF | 78.00% |
| GRU | TF-IDF | 92.00% |
| Bi-Directional RNN | TF-IDF | **94.00%** |
| Bi-Directional LSTM | TF-IDF | 87.00% |
| Bi-Directional GRU | TF-IDF | **93.00%** |
| Transformer | TF-IDF | 90.00% |
| Transformer | Word2Vec | 91.00% |

## Analysis and Key Findings

### Impact of Text Representation
- **TF-IDF consistently outperforms Word2Vec across most models.**
- **Word2Vec may lose discriminative word importance when aggregating vectors for entire documents.**

### Model Performance
- **MLP with TF-IDF is the top-performing model (94.16%).**
- **Bidirectional RNN and GRU with TF-IDF also achieve high accuracy (94% and 93%).**
- **Transformers show competitive performance but do not surpass MLP or bidirectional RNN.**
- **LSTM with TF-IDF underperforms (78%) due to its mismatch with non-sequential TF-IDF representations.**

## Conclusion and Recommendations

### Conclusion
- **TF-IDF is a better feature extraction method for fake news classification.**
- **MLP (94.16%) and Bidirectional RNN (94%) provide the best performance.**
- **Transformer models require further tuning but show promising results.**
- **Finetuned - BERT(Bidirectional Encoder Representations from Transformers )** which outperformed all the models.

### Recommendations
- **Experiment with BERT or other contextual embeddings.**
- **Combine TF-IDF and Word2Vec in ensemble methods.**
- **Apply data augmentation techniques to improve model robustness.**

## License
This project is open-source and available under the [MIT License](LICENSE).

