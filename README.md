# CT Classification Project

A deep learning project for classifying CT scan images, aimed at improving model accuracy through K-Fold cross-validation and robust performance evaluation metrics. The pipeline includes data preprocessing, model training with TensorFlow, and evaluation using accuracy and ROC AUC metrics. Visualizations, such as confusion matrices, precision-recall curves, and ROC curves, provide insights into model performance across folds.

## Project Structure
- **`train_and_evaluate.py`** - Manages model training and evaluation with cross-validation.
- **`models.py`** - Defines the architecture of the classification model.
- **`load_data.py`** - Handles data loading and preprocessing.
- **`preprocess_and_label_data.py`** - Preprocesses raw data and assigns labels.
- **`processed_data`** - Folder containing the preprocessed dataset used for training.

## Dependencies
Ensure you have all dependencies installed from `requirements.txt` for a smooth setup.
