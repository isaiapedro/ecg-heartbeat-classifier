# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Deep Learning
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Conv1D, MaxPooling1D, Flatten

import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(y_test_classes, y_pred_test_classes):
    accuracy = accuracy_score(y_test_classes, y_pred_test_classes)
    precision = precision_score(y_test_classes, y_pred_test_classes, average='micro')
    recall = recall_score(y_test_classes, y_pred_test_classes, average='micro')
    f1 = f1_score(y_test_classes, y_pred_test_classes, average='micro')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3}")
    print(f"F1-score: {f1:.3f}")

def randomForest(X_train_normalized, X_valid_normalized, y_train, y_valid):
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=123)

    rf_classifier.fit(X_train_normalized, y_train)

    y_pred_valid = rf_classifier.predict(X_valid_normalized)
    accuracy_valid = accuracy_score(y_valid, y_pred_valid)

    return rf_classifier

def ann(X_train_normalized, X_valid_normalized, y_train, y_valid):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    ann_model = Sequential([
        Input(shape=(X_train_normalized.shape[1],)),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(5, activation="softmax")
    ])

    ann_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    plot_model(ann_model, to_file='Ann_model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

    history_ann = ann_model.fit(
       X_train_normalized, y_train,
        validation_data=(X_valid_normalized, y_valid),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_ann.history['accuracy'], label='Training Accuracy')
    plt.plot(history_ann.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_ann.history['loss'], label='Training Loss')
    plt.plot(history_ann.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ann_model

def cnn(X_train_normalized, X_valid_normalized, y_train, y_valid):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    cnn_model = Sequential([
        Input(shape=(X_train_normalized.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(5, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    plot_model(cnn_model, to_file='CNN_model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

    history_cnn = cnn_model.fit(
        X_train_normalized.reshape(-1, X_train_normalized.shape[1], 1),
        y_train,
        validation_data=(X_valid_normalized.reshape(-1, X_valid_normalized.shape[1], 1), y_valid),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
    plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_cnn.history['loss'], label='Training Loss')
    plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return cnn_model

def lstm(X_train_normalized, X_valid_normalized, y_train, y_valid):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    lstm_model = Sequential([
        Input(shape=(X_train_normalized.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(5, activation='softmax')
    ])

    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    plot_model(lstm_model, to_file='RNN_model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

    history_rnn = lstm_model.fit(
        X_train_normalized.reshape(-1, X_train_normalized.shape[1],1),
        y_train,
        validation_data=(X_valid_normalized.reshape(-1, X_valid_normalized.shape[1],1), y_valid),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
    plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_rnn.history['loss'], label='Training Loss')
    plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return lstm_model

def ensemble_nn(X_test_normalized, y_test):

    ann_model = tf.keras.models.load_model("ANN_model.h5")

    cnn_model = tf.keras.models.load_model("CNN_model.h5")

    lstm_model = tf.keras.models.load_model("LSTM_model.h5")

    ann_probs = ann_model.predict(X_test_normalized)

    cnn_probs = cnn_model.predict(X_test_normalized.reshape(-1, 187, 1))

    lstm_probs = lstm_model.predict(X_test_normalized.reshape(-1, X_test_normalized.shape[1], 1))

    ensemble_probs = (ann_probs + cnn_probs + lstm_probs) / 3

    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    ensemble_accuracy = np.mean(ensemble_preds == y_test)
    print("Ensemble Accuracy:", ensemble_accuracy)

    ann_probs_vot = ann_model.predict(X_test_normalized)
    cnn_probs_vot = cnn_model.predict(X_test_normalized.reshape(-1, 187, 1))
    rnn_probs_vot = lstm_model.predict(X_test_normalized.reshape(-1, 187, 1))

    ann_preds = np.argmax(ann_probs_vot, axis=1)
    cnn_preds = np.argmax(cnn_probs_vot, axis=1)
    rnn_preds = np.argmax(rnn_probs_vot, axis=1)

    ensemble_preds = []

    for i in range(len(X_test_normalized)):
        votes = [ann_preds[i], cnn_preds[i], rnn_preds[i]]
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_preds.append(ensemble_pred)

    ensemble_preds = np.array(ensemble_preds)
    ensemble_accuracy = np.mean(ensemble_preds == y_test)
    print("Ensemble Accuracy:", ensemble_accuracy)

    return ensemble_preds