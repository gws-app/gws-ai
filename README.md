# Multi-Model NLP Repository

This repository contains two distinct models focused on Natural Language Processing (NLP):

1. **FAQ Chatbot Model**
   - A chatbot that uses deep learning techniques to classify user inputs and provide appropriate responses.
   - The model employs preprocessing, stemming, and tokenization techniques specific to the Indonesian language.
   - Built with TensorFlow and Keras, the chatbot supports:
     - Training with K-fold cross-validation.
     - Fine-tuning using early stopping.
     - Conversion to TensorFlow Lite for deployment.

2. **Activity Recommendation System**
   - A recommendation engine for suggesting activities based on user input and a dataset of activities.
   - Uses TF-IDF vectorization and cosine similarity to determine the most relevant recommendations.
   - Designed for high accuracy and performance in text-based similarity analysis.

## Features

### FAQ Chatbot Model
- **Preprocessing**: Tokenization, stemming (using PySastrawi), and normalization for effective input handling.
- **Model Architecture**:
  - Input and hidden layers with dropout and batch normalization.
  - Softmax output for multi-class classification.
- **Cross-Validation**: Implements K-fold cross-validation to validate performance across multiple folds.
- **Deployment**: Converts the trained model to TensorFlow Lite format for lightweight applications.

### Activity Recommendation System
- **Vectorization**: Uses TF-IDF vectorizer to transform activities into feature vectors.
- **Similarity Calculation**: Applies cosine similarity to rank and recommend activities.
- **Custom Tokenization**: Handles flexible and non-standardized input activity data.
- **User-Friendly Recommendations**: Filters out duplicate activities from recommendations and provides clean outputs.

## Installation

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

Dependencies include:
- TensorFlow
- NLTK
- PySastrawi
- NumPy
- Pandas
- Scikit-learn

## Dataset
- The chatbot model requires an intents dataset (`nlp_dataset_faq.json`) formatted with patterns, responses, and tags.
- The activity recommendation model uses a CSV file (`clean-dataset.csv`) with activity data.

## Usage

### FAQ Chatbot Model
1. Train the model by running the notebook or script.
2. Save the trained model and metadata (`words`, `classes`, etc.) for future use.
3. Test the chatbot by entering user inputs and observing responses.

Example:
```python
message=("Bagaimana cara mengatasi kesedihan?")
```

### Activity Recommendation System
1. Load the activity dataset and preprocess it.
2. Generate the TF-IDF matrix and calculate cosine similarity.
3. Use the recommendation function to get activity suggestions.

Example:
```python
input_activities = "walking | fasting | youtube"
result = recommend_activities(input_activities, df, vectorizer, tfidf_matrix, top_n=5)
print(result)
```

## Saved Models and Artifacts
- `final_model_trained_on_full_data.h5`: Trained chatbot model.
- `model.tflite`: TensorFlow Lite version of the chatbot model.
- `vectorizer.pkl`: TF-IDF vectorizer for the recommendation system.
- `tfidf_matrix.pkl`: Precomputed TF-IDF matrix.
- `dataset.pkl`: Serialized activity dataset.


## License
Please read [LICENSE](https://github.com/gws-app/gws-ai/blob/main/LICENSE) file for details.

## Acknowledgments
- The chatbot uses PySastrawi for stemming and NLTK for tokenization.
- The recommendation engine is inspired by best practices in text similarity analysis.

