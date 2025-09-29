# Email Spam Classifier - End-to-End Machine Learning Project

## 📹 Project Demo
Watch the full project demonstration: [Demo Video](https://drive.google.com/file/d/1IO-_aNMk5cW5MZ-k9UZ0lyXKM7y4TASh/view?usp=sharing)

## 📄 View Code
- **PDF Format**: [View Notebook PDF](https://github.com/LearnCode801/Email-Spam-Ham-Classifier-ML-Project/blob/main/End-to-End%20Project%20of%20Email%20Spam%20Classifier.pdf)
- **Jupyter Notebook**: [View Interactive Notebook](https://github.com/LearnCode801/Email-Spam-Ham-Classifier-ML-Project/blob/main/End-to-End%20Project%20of%20Email%20Spam%20Classifier.ipynb)

## Project Overview
This project builds a comprehensive email spam classification system using multiple machine learning algorithms. The classifier distinguishes between legitimate emails (ham) and spam emails with high accuracy.

## Dataset
- **Source**: SMS Spam Collection Dataset
- **Total Records**: 5,572 emails
- **Features**: Email text content and binary labels (ham/spam)
- **After Preprocessing**: 5,169 unique emails

## Project Workflow

### 1. Data Cleaning
- Removed unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renamed columns to `email` and `target`
- Applied label encoding: Ham=0, Spam=1
- Handled missing values (none found)
- Removed 403 duplicate emails
- Reset index for clean dataset

### 2. Exploratory Data Analysis (EDA)

#### Class Distribution
- **Ham (legitimate)**: 4,516 emails (87.4%)
- **Spam**: 653 emails (12.6%)
- Dataset shows class imbalance favoring ham emails

#### Feature Engineering
Created three new numerical features:
- `num_char`: Character count per email
- `num_words`: Word count per email
- `num_sent`: Sentence count per email

#### Statistical Insights

**Ham Emails:**
- Average characters: 70.5
- Average words: 17.1
- Average sentences: 1.8

**Spam Emails:**
- Average characters: 137.9
- Average words: 27.7
- Average sentences: 3.0

**Key Finding**: Spam emails are significantly longer than ham emails in all metrics.

#### Word Frequency Analysis

**Top Spam Keywords:**
- call (320), free (191), txt (130), mobile (105), claim (98), prize (82), urgent (57), cash (51)

**Top Ham Keywords:**
- u (883), get (588), go (495), come (298), know (247), like (231), good (212)

### 3. Text Preprocessing

Applied comprehensive NLP preprocessing:
1. **Lowercasing**: Standardized all text
2. **Tokenization**: Split text into words using NLTK
3. **Alphanumeric Filtering**: Kept only alphanumeric tokens
4. **Stopword Removal**: Removed common English words
5. **Punctuation Removal**: Cleaned special characters
6. **Lemmatization**: Reduced words to base form using WordNet

**Example Transformation:**
- Input: `"jiood i am talha is laughing %%% & 7 AAA ALI"`
- Output: `"jiood talha laugh 7 aaa ali"`

### 4. Text Vectorization

Tested two vectorization methods:

#### CountVectorizer
- Converts text to token count matrix
- Output shape: (5,169, 7,055)
- Simple frequency-based approach

#### TfidfVectorizer (Selected)
- Term Frequency-Inverse Document Frequency weighting
- Max features: 3,000
- Output shape: (5,169, 3,000)
- Better handles common vs. distinctive words

### 5. Model Building

#### Train-Test Split
- Training set: 4,135 emails (80%)
- Test set: 1,034 emails (20%)
- Random state: 2 (for reproducibility)

#### Models Evaluated (13 algorithms)

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| Bernoulli Naive Bayes | 98.36% | 99.19% |
| Extra Trees Classifier | 97.87% | 98.33% |
| Support Vector Classifier | 97.58% | 97.48% |
| Random Forest | 97.20% | 97.39% |
| Multinomial Naive Bayes | 97.20% | 100.00% |
| XGBoost | 96.91% | 95.69% |
| AdaBoost | 95.94% | 94.44% |
| Bagging Classifier | 95.65% | 86.05% |
| Logistic Regression | 95.36% | 95.92% |
| Gradient Boosting | 94.97% | 94.79% |
| Decision Tree | 93.23% | 85.42% |
| K-Nearest Neighbors | 90.52% | 100.00% |
| Gaussian Naive Bayes | 86.46% | 49.56% |

### 6. Ensemble Methods

#### Voting Classifier
- Combination: SVC + Multinomial NB + Extra Trees
- Voting type: Soft voting (probability-based)
- **Accuracy**: 98.36%
- **Precision**: 100.00%

#### Stacking Classifier
- Base estimators: SVC, Multinomial NB, Extra Trees, Bernoulli NB
- Meta estimator: Random Forest
- **Accuracy**: 98.36%
- **Precision**: 96.90%

### 7. Model Deployment

All trained models saved using pickle for production deployment:
- 13 individual classifiers
- 1 TF-IDF vectorizer
- 2 ensemble models (voting & stacking)

Total of 16 pickle files created for flexible model selection.

## Key Results

### Best Performing Models
1. **Bernoulli Naive Bayes**: Best balance of accuracy (98.36%) and precision (99.19%)
2. **Voting Ensemble**: Highest precision (100%) with excellent accuracy
3. **Extra Trees**: Strong performance (97.87% accuracy, 98.33% precision)

### Model Selection Criteria
- **For Production**: Voting Classifier (perfect precision prevents false spam classifications)
- **For Speed**: Bernoulli Naive Bayes (fast, lightweight, excellent performance)
- **For Robustness**: Stacking Classifier (handles diverse patterns well)

## Technical Stack

### Libraries Used
- **Data Processing**: pandas, numpy
- **NLP**: nltk (tokenization, lemmatization, stopwords)
- **Vectorization**: sklearn.feature_extraction.text
- **Modeling**: sklearn, xgboost
- **Visualization**: matplotlib, seaborn, wordcloud
- **Persistence**: pickle

### System Requirements
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
nltk>=3.6
xgboost>=1.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

## Future Improvements

1. **Handle Class Imbalance**: Apply SMOTE or class weighting
2. **Deep Learning**: Experiment with LSTM/BERT models
3. **Feature Engineering**: Add metadata features (sender, time, attachments)
4. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
5. **Web Interface**: Deploy with Streamlit or Flask
6. **Real-time Processing**: Stream processing for live email filtering
7. **Multi-language Support**: Extend beyond English emails

## Usage Example

```python
import pickle

# Load vectorizer and model
vectorizer = pickle.load(open('email_spam_vectorizer.pkl', 'rb'))
model = pickle.load(open('voting_email_spam_model.pkl', 'rb'))

# Preprocess and predict
def predict_spam(email_text):
    # Apply same preprocessing as training
    processed_text = transform_text(email_text)
    vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test
email = "Congratulations! You've won a free prize. Call now!"
print(predict_spam(email))  # Output: Spam
```

## Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline for email spam classification. With 98.36% accuracy and 100% precision using the voting ensemble, the system reliably identifies spam while minimizing false positives. The modular design allows for easy model updates and deployment in production environments.
