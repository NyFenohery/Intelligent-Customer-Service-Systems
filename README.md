# NLP Solutions for Real-World Problems
## Intelligent Customer Service Systems

**COMP8460 - Assignment 3**

### Team Members
- Khaliun Munkh-Ochir | 48715409
- Ny Fenohery Jeannot | 48996637
- Aashish Tomar | 48396214

---

## 📋 Project Overview

This project implements an intelligent customer service system using Natural Language Processing (NLP) techniques. The system processes IT service tickets, performs sentiment analysis, classifies requests, and generates automated responses using both traditional machine learning and modern LLM approaches.

### Key Features

- **Text Preprocessing**: Tokenization, stopword removal, and lemmatization
- **POS Tagging**: Understanding query structure using spaCy
- **Named Entity Recognition (NER)**: Customer and product information extraction
- **Text Classification**: Ticket categorization using Naive Bayes and Logistic Regression
- **Sentiment Analysis**: Multi-method approach using VADER and TextBlob
- **Response Generation**: Automated reply generation using Qwen LLM
- **RAG Implementation**: Retrieval-Augmented Generation for company policy matching
- **Dialogue State Tracking**: Intent and entity tracking across conversations
- **Escalation System**: Rule-based and LLM-based escalation detection

---

## 🗂️ Dataset

**Source**: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

The dataset contains IT service tickets with multiple fields including:
- Document text (ticket descriptions)
- Topic groups (categories)
- Various metadata fields

**Note**: The project uses a subset of 1000 tickets for computational efficiency.

---

## 🛠️ Technologies & Libraries

### Core NLP Libraries
- **NLTK**: Tokenization, stopwords, lemmatization
- **spaCy**: POS tagging and Named Entity Recognition
- **Transformers**: Hugging Face library for LLM integration

### Machine Learning
- **scikit-learn**: Classification models, TF-IDF vectorization, metrics
- **PyTorch**: Deep learning framework

### Sentiment Analysis
- **VADER**: Rule-based sentiment analysis
- **TextBlob**: Alternative sentiment analysis

### LLM
- **Qwen (3-0.6B)**: Lightweight language model for response generation

### Data Processing & Visualization
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib & seaborn**: Data visualization

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster LLM inference)

### Install Dependencies

```bash
pip install -q wordcloud textblob vaderSentiment
pip install nltk spacy torch pandas numpy matplotlib seaborn kagglehub
pip install transformers datasets scikit-learn
```

### Download Required Resources

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## 📊 Pipeline Architecture

### 1. **Preprocessing**
- Text tokenization using NLTK
- Lowercase conversion
- Punctuation removal
- Stopword filtering
- Lemmatization with WordNetLemmatizer

### 2. **Feature Extraction**
- **POS Tagging**: Extracts NOUN and VERB tags for meaningful content
- **NER**: Identifies CARDINAL and DATE entities
- **TF-IDF**: Creates document-term matrix for similarity computation

### 3. **Classification**
Two models are trained for ticket classification:
- **Naive Bayes**: Fast baseline model
- **Logistic Regression**: Improved accuracy with max_iter=1000

### 4. **Sentiment Analysis**
Dual approach for robustness:
- **VADER**: Compound score-based classification (thresholds: ±0.675)
- **TextBlob**: Polarity-based sentiment detection

### 5. **Response Generation**
- **Prompt Engineering**: Few-shot learning with instruction-based prompts
- **RAG System**: Policy retrieval using TF-IDF similarity
- **LLM Generation**: Qwen model generates contextual replies

### 6. **Advanced Features**

#### Dialogue State Tracking
Tracks conversation state with:
- Intent detection (from ticket category)
- Entity extraction (order IDs, accounts)
- Stage management (start → ask_info → resolving → done)

#### Escalation Detection
Dual-layer approach:
- **Rule-based**: Keyword matching + extreme negative sentiment
- **LLM-based**: Contextual understanding via Qwen

---

## 💻 Usage

### Running the Notebook

Open `Assignment_3.ipynb` in Jupyter or VS Code and execute cells sequentially.

### Example: Full Pipeline

```python
# Process a customer review
review_text = "My order arrived late and nobody replied to my emails."

result = full_pipeline_with_escalation(review_text)

print(f"Sentiment: {result['sentiment']}")
print(f"Policy: {result['policy']}")
print(f"Reply: {result['reply']}")
print(f"Escalate: {result['final_escalate']}")
```

### Example Output
```
Sentiment: Negative
Policy: If delivery issues occur, apologise and offer tracking support or replacement.
Reply: We're really sorry for the delay and lack of communication. Please share your order details so we can resolve this quickly.
Escalate: False
```

---

## 📈 Model Performance

### Classification Results

#### Naive Bayes
- Fast training and prediction
- Suitable for baseline text classification
- Performance metrics available in notebook

#### Logistic Regression
- Improved accuracy over Naive Bayes
- Better handling of feature correlations
- Detailed classification report and confusion matrix provided

### Sentiment Analysis

The system uses VADER as the primary sentiment analyzer with TextBlob for comparison. VADER performs better on informal text and handles negations effectively.

---

## 🔍 Key Insights

1. **POS + NER Filtering**: Focusing on NOUN/VERB tags and specific entity types (CARDINAL, DATE) improved feature quality while reducing noise from mislabeled entities.

2. **TF-IDF Sparsity**: High sparsity (>99%) indicates low document similarity, which is expected for diverse ticket types.

3. **Sentiment Thresholds**: Custom VADER threshold (±0.675) provides better separation between neutral and polar sentiments.

4. **RAG Effectiveness**: Policy retrieval significantly improves response relevance by grounding LLM outputs in company guidelines.

5. **Escalation Accuracy**: Combining rule-based and LLM-based approaches reduces false positives while catching edge cases.

---

## 📁 Project Structure

```
AI in NLP/
│
├── Assignment_3.ipynb    # Main notebook with complete implementation
└── README.md            # This file
```

---

## 🔮 Future Improvements

- [ ] Fine-tune Qwen model on customer service data
- [ ] Expand policy knowledge base with real company documents
- [ ] Implement multi-turn dialogue management
- [ ] Add support for multiple languages
- [ ] Deploy as REST API with FastAPI
- [ ] Integrate feedback loop for continuous improvement
- [ ] Add A/B testing framework for response quality

---

## 📚 References

- [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Qwen Model](https://huggingface.co/Qwen)
- [spaCy NLP Library](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 📝 License

This project is submitted as part of academic coursework for COMP8460.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the team members listed above.
