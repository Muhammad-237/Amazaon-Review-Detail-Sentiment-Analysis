# Amazon Review Detailed Sentiment Analysis

## Overview

This project focuses on sentiment analysis of Amazon reviews using a public dataset from Kaggle. The aim was to classify reviews as positive or negative by implementing various machine learning algorithms and selecting the best-performing model. Detailed analysis and preprocessing steps were conducted before model training to ensure robust results.

## Project Workflow

1. **Data Collection and Exploration:**
   - Sourced Amazon review data from Kaggle.
   - Conducted an in-depth exploratory data analysis (EDA) to understand the distribution and characteristics of the dataset.

2. **Data Preprocessing:**
   - Cleaned and prepared the dataset by handling missing values, normalizing text, and performing tokenization.
   - Converted text data into numerical features using techniques like TF-IDF and word embeddings.

3. **Model Implementation:**
   - Implemented multiple machine learning algorithms:
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Naive Bayes

4. **Model Selection:**
   - Evaluated the performance of each model using accuracy, precision, recall, and F1-score.
   - Selected Logistic Regression as the best-performing model with an initial accuracy of 88%.

5. **Hyperparameter Tuning:**
   - Performed hyperparameter tuning on Logistic Regression using techniques such as Grid Search and Cross-Validation.
   - Achieved an improved accuracy of 95% after tuning.

## Results

- **Best Model:** Logistic Regression
- **Initial Accuracy:** 88%
- **Final Accuracy after Tuning:** 95%

## Conclusion

The project successfully demonstrated the process of sentiment analysis on Amazon reviews, from data exploration and preprocessing to model implementation and optimization. The Logistic Regression model, after hyperparameter tuning, provided a highly accurate classification of review sentiments.

## Repository Contents

- **data/**: Contains the original and processed datasets.
- **notebooks/**: Jupyter notebooks detailing the data exploration, preprocessing steps, model training, and evaluation.
- **models/**: Saved models and hyperparameter tuning configurations.
- **results/**: Performance metrics and visualizations.

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/amazon-review-sentiment-analysis.git
   cd amazon-review-sentiment-analysis
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks to see the analysis and model training steps:
   ```sh
   jupyter notebook
   ```

## Future Work

- Explore deep learning models such as LSTM and BERT for potential performance improvement.
- Implement real-time sentiment analysis on streaming data.
- Expand the analysis to include multi-class sentiment classification (e.g., positive, negative, neutral).

## Contact

For any questions or suggestions, please reach out to [your email] or open an issue on GitHub.

---

Feel free to customize this README file further to suit your needs!
