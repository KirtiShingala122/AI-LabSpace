# AI Lab Experiments

A collection of practical AI and Machine Learning implementations covering regression, classification, deep learning, NLP, recommendation systems, and reinforcement learning.

## Repository Structure

```
AI_lab_experiments/
├── Exp-2_liniar_regression/
│   ├── Exp-2_liniar_regression.ipynb
│   └── exp-1_train.csv
├── Exp-3_KNN_classification_algorithm.ipynb
├── Exp-4_CNN.ipynb
├── Exp-6_basics_of_NLP_also_summurization.ipynb
├── Exp-7_Ngram_Word_Prediction.ipynb
├── Exp-8_Word_Embedding.ipynb
├── Exp-9_TextRank_for_keyword_extraction.ipynb
├── Exp-10_Text_Rank_Summarization.ipynb
├── Exp-11&12_moive_recommendation(...).ipynb
├── Exp-13_Reinforcement_Learning.ipynb
└── README.md
```

---

## Experiments

### 1. Linear Regression

**File:** `Exp-2_liniar_regression/Exp-2_liniar_regression.ipynb`

Simple linear regression implementation on a training dataset:

- Loading and exploring CSV data with Pandas
- Statistical summary using `describe()`
- Feature and target variable extraction
- Model fitting and coefficient estimation
- Scatter plot visualization with regression line

**Key Libraries:** NumPy, Pandas, Matplotlib, scikit-learn

---

### 2. KNN Classification

**File:** `Exp-3_KNN_classification_algorithm.ipynb`

From-scratch K-Nearest Neighbors implementation on the Iris dataset:

- Loading Iris data directly from UCI repository
- Custom Euclidean distance function
- Distance computation between test sample and all training points
- Sorting by distance and selecting top-k neighbors
- Majority voting using Counter for class prediction
- Comparing predicted vs actual labels

**Key Libraries:** NumPy, Pandas, Matplotlib, math, collections

**Approach:** Manual implementation without library classifiers to demonstrate core KNN mechanics

---

### 3. Convolutional Neural Network (CNN)

**File:** `Exp-4_CNN.ipynb`

CNN for image classification on the Fashion-MNIST dataset:

- Data loading, reshaping, and normalization (0-1 scaling)
- One-hot encoding of labels
- Train-validation split (80-20)
- Three convolutional blocks (Conv2D + LeakyReLU + MaxPooling2D)
- Filter progression: 32 -> 64 -> 128
- Flatten + Dense(128) + Softmax output for 10 classes
- Training for 20 epochs with Adam optimizer
- Test accuracy evaluation (~92%)
- Accuracy and loss curve visualization

**Key Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, scikit-learn

---

### 4. NLP Fundamentals and Text Summarization

**File:** `Exp-6_basics_of_NLP_also_summurization.ipynb`

Core NLP techniques combined with extractive text summarization:

- Text tokenization and word splitting
- Named Entity Recognition (NER) using spaCy
- Part-of-Speech (POS) tagging
- Stopword identification and filtering
- Stemming with Porter Stemmer
- Lemmatization using spaCy for context-aware normalization
- TextRank-based document summarization using sentence similarity and PageRank

**Key Libraries:** spaCy, NLTK, NetworkX, NumPy, Pandas

---

### 5. N-gram Word Prediction

**File:** `Exp-7_Ngram_Word_Prediction.ipynb`

Bigram-based language model for next word prediction:

- Text tokenization using NLTK
- Bigram generation from corpus
- Frequency table construction with Counter and defaultdict
- Prediction function returning the most probable next word
- Testing with various input words

**Key Libraries:** NLTK, collections

**Approach:** Statistical model using maximum likelihood estimation over bigram frequencies

---

### 6. Word Embedding and Document Similarity

**File:** `Exp-8_Word_Embedding.ipynb`

Text vectorization and similarity measurement:

- Bag-of-Words representation using CountVectorizer
- TF-IDF vectorization using TfidfVectorizer
- Cosine similarity computation between document pairs
- Custom function to find the most similar document for each sentence

**Key Libraries:** scikit-learn (CountVectorizer, TfidfVectorizer, cosine_similarity), NumPy, Pandas

**Approach:** Lexical similarity based on TF-IDF weighted term overlap

---

### 7. TextRank for Keyword Extraction

**File:** `Exp-9_TextRank_for_keyword_extraction.ipynb`

Graph-based keyword extraction using the TextRank algorithm:

- Text preprocessing (tokenization, lowercasing, stopword removal)
- Co-occurrence graph construction within a sliding window
- PageRank algorithm for word importance scoring
- Top-N keyword extraction

**Key Libraries:** NLTK, NetworkX

---

### 8. TextRank for Document Summarization

**File:** `Exp-10_Text_Rank_Summarization.ipynb`

Sentence-level extractive summarization using TextRank:

- Sentence tokenization and preprocessing
- Pairwise sentence similarity matrix construction
- Graph creation from similarity matrix
- PageRank scoring to rank sentences
- Top-N sentence selection as summary output

**Key Libraries:** NumPy, Pandas, NLTK, NetworkX, Matplotlib

---

### 9. Movie Recommendation System (Content-Based and Collaborative Filtering)

**File:** `Exp-11&12_moive_recommendation(content and collabrative based filtering).ipynb`

Two recommendation approaches in a single notebook:

**Content-Based Filtering:**
- Combining movie genres and overviews as content features
- TF-IDF vectorization of content
- Cosine similarity matrix between all movies
- Top-N similar movie recommendations by title

**Collaborative Filtering:**
- User-item rating matrix creation using pivot table
- User-to-user cosine similarity computation
- Weighted rating prediction for unwatched movies
- Top-5 movie recommendations for a target user

**Key Libraries:** Pandas, scikit-learn (TfidfVectorizer, cosine_similarity), NumPy

**Dataset:** Requires `movies.csv` and `ratings.csv`

---

### 10. Reinforcement Learning (Q-Learning)

**File:** `Exp-13_Reinforcement_Learning.ipynb`

Q-Learning algorithm for optimal path finding in a graph environment:

- Graph creation and visualization using NetworkX
- Reward matrix definition with goal state reward of 100
- Q-matrix initialization and iterative updates
- Q-learning update rule: `Q(s,a) = R(s,a) + gamma * max(Q(a,:))`
- Training over 1000 episodes with gamma = 0.75
- Convergence monitoring through Q-matrix prints

**Key Libraries:** NumPy, NetworkX, Matplotlib

---

## Setup

### Requirements
- Python 3.7+
- Jupyter Notebook or Google Colab

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras nltk spacy networkx
python -m spacy download en_core_web_sm
```

### NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| ML / DL | scikit-learn, TensorFlow, Keras |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK, spaCy |
| Graphs | NetworkX |
| Environment | Jupyter Notebook, Google Colab |

---

## Author

Kirti Shingala

## License

This repository is open for educational purposes.
