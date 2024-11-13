"""
------------------------------------------------

File: train_and_evaluate.py
Author: Noman Kazi
Date: 2024-11-13

Description: Script to manage the model training and evaluation process. 
             Includes data loading, setting up K-Fold cross-validation, 
             tracking accuracy and ROC AUC scores, and generating performance plots.
             
------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from load_data import load_data
from models import create_model
from tensorflow.keras import mixed_precision

# Ensure GPU is being used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth to avoid memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU detected. Check your runtime configuration.")

# Enable mixed precision to optimize GPU usage
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Paths
label_path = r"C:\Users\noman\Downloads\CTClassification\processed_data\label.txt"
data_path = r"C:\Users\noman\Downloads\CTClassification\processed_data"

# Load data and print shapes for confirmation
print("Starting data loading...")
labels, data_x, data_y, data_z = load_data(label_path, data_path)
print("Data loading complete.")
print(f"Data shapes - X: {data_x.shape}, Y: {data_y.shape}, Z: {data_z.shape}, Labels: {labels.shape}")

class EpochEndCallback(Callback):
    def __init__(self, display_interval=10):
        super().__init__()
        self.display_interval = display_interval
        self.epoch_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy", 0) * 100
        self.epoch_accuracies.append(accuracy)
        if (epoch + 1) % self.display_interval == 0:
            val_accuracy = logs.get("val_accuracy", 0) * 100
            loss = logs.get("loss", 0) * 100
            val_loss = logs.get("val_loss", 0) * 100
            print(f"Epoch {epoch + 1}: Accuracy: {accuracy:.2f}%, Loss: {loss:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.2f}%")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
fold_results = []

for train_index, val_index in kf.split(data_x):
    print(f"\nTraining fold {fold}...")

    X_train = [data_x[train_index], data_y[train_index], data_z[train_index]]
    X_val = [data_x[val_index], data_y[val_index], data_z[val_index]]
    y_train, y_val = labels[train_index], labels[val_index]

    model = create_model()
    checkpoint = ModelCheckpoint(f"best_model_fold_{fold}.keras", save_best_only=True, monitor="val_loss", mode="min")
    epoch_callback = EpochEndCallback(display_interval=10)
    
    with tf.device('/GPU:0'):  # Explicitly use GPU
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=8,
            callbacks=[checkpoint, epoch_callback],
            verbose=0
        )

    # Predictions and Metrics Calculation
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    accuracy = accuracy_score(y_val, y_pred) * 100
    try:
        roc_auc = roc_auc_score(y_val, y_pred_prob) * 100
    except ValueError:
        roc_auc = float('nan')
    avg_epoch_accuracy = np.mean(epoch_callback.epoch_accuracies)
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'confusion_matrix_fold_{fold}.png')
    plt.close()

    # Precision-Recall Curve Plot
    precision_values, recall_values, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_auc = auc(recall_values, precision_values)
    plt.figure()
    plt.plot(recall_values, precision_values, label=f'Fold {fold} (AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(f'precision_recall_curve_fold_{fold}.png')
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(f'roc_curve_fold_{fold}.png')
    plt.close()

    # Append Results for this Fold
    fold_results.append((fold, accuracy, roc_auc, avg_epoch_accuracy))
    print(f"Fold {fold} - Accuracy: {accuracy:.2f}%, ROC AUC: {roc_auc if not np.isnan(roc_auc) else 'Undefined'}%, Average Epoch Accuracy: {avg_epoch_accuracy:.2f}%")
    fold += 1

# Print Summary of Fold Results
print("\nSummary of Fold Results:")
print("Fold\tAccuracy (%)\tROC AUC (%)\tAvg Epoch Accuracy (%)")
for fold, accuracy, roc_auc, avg_epoch_accuracy in fold_results:
    print(f"{fold}\t{accuracy:.2f}\t\t{roc_auc if not np.isnan(roc_auc) else 'Undefined'}\t\t{avg_epoch_accuracy:.2f}")

# Average Metrics across Folds
average_accuracy = np.mean([result[1] for result in fold_results])
average_roc_auc = np.nanmean([result[2] for result in fold_results])
average_epoch_accuracy = np.mean([result[3] for result in fold_results])
print(f"\nAverage Accuracy across all folds: {average_accuracy:.2f}%")
print(f"Average ROC AUC across all folds: {average_roc_auc if not np.isnan(average_roc_auc) else 'Undefined'}%")
print(f"Average Epoch Accuracy across all folds: {average_epoch_accuracy:.2f}%")

