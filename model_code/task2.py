# Should be the same as task1 but only difference is that it will need 
# to classify the static activities with the classifiable breathing
# Hence:
# class 0: sitting/standing + breathing normally
# class 1: lying down on your left side + breathing normally
# class 2: lying down on your right side + breathing normally
# class 3: lying down on your back + breathing normally
# class 4: lying down on your stomach + breathing normally 
# class 5: sitting/standing + coughing
# class 6: lying down on your left side + coughing
# class 7: lying down on your right side + coughing
# class 8: lying down on your back + coughing
# class 9: lying down on your stomach + coughing
# class 10: sitting/standing + hyperventilating
# class 11: lying down on your left side + hyperventilating
# class 12: lying down on your right side + hyperventilating
# class 13: lying down on your back + hyperventilating
# class 14: lying down on your stomach + hyperventilating

import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, GRU, MaxPooling1D, GlobalMaxPool1D, Activation,LSTM
from keras.layers import Bidirectional
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import Model
from keras.layers import Input
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


# def create_segments_and_labels(df, window_size, overlap_size):
#     segments = []
#     p_labels = []
#     b_labels = []

#     for file in df["activity"].unique():
#         file_df = df[df["activity"] == file].drop(columns="activity")

#         start = 0
#         while start < len(file_df):
#             end = start + window_size
#             # Ensure not to include data beyond the current activity
#             if end > len(file_df):
#                 break
#             segment = file_df.iloc[start:end]
#             if len(segment) == window_size:  # Ensure the segment is full
#                 segments.append(segment.iloc[:, :-2].values)  # Exclude the label column
#                 # Assign label based on the activity at the start of the window
#                 p_labels.append(segment["physical_label"].values[0])
#                 b_labels.append(segment["activity_label"].values[0])
#             start += int(window_size * (1 - overlap_size))  # Move the window

#     return np.array(segments), np.array(p_labels), np.array(b_labels)

def create_segments_and_labels_2(df, window_size, overlap_size, label_col='activity_label'):
    segments = []
    labels = []
    df = df.drop(columns="physical_label")
    for activity in df[label_col].unique():
        activity_df = df[df[label_col] == activity]
        start = 0
        while start < len(activity_df):
            end = start + window_size
            # Ensure not to include data beyond the current activity
            if end > len(activity_df):
                break
            segment = activity_df.iloc[start:end]
            if len(segment) == window_size:  # Ensure the segment is full
                segments.append(segment.iloc[:, :-1].values)  # Exclude the label column
                # Assign label based on the activity at the start of the window
                labels.append(segment[label_col].values[0])
            start += int(window_size * (1 - overlap_size))  # Move the window
    return np.array(segments), np.array(labels)


def create_segments_and_labels(df, window_size, step_size, target_col, drop_col):
    segments = []
    labels = []
    df = df.drop(columns=drop_col)
    for start in range(0, len(df) - window_size, step_size):
        segment = df.iloc[start:start+window_size]
        label = mode(segment[target_col])[0]
        segments.append(segment)
        labels.append(label)
    
    return np.array(segments), np.array(labels)


def train_model_1(X_train_shape,y_train_shape):
    model_architecture = tf.keras.Sequential([
       Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train_shape[1], X_train_shape[2])),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),

        Conv1D(filters=256, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),

        GlobalMaxPool1D(),

        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(y_train_shape[1], activation='softmax')
    ])
    # Compile the model
    model_architecture.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_architecture


def train_model_2(X_train_shape,y_train_shape):
    model = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=5, input_shape=(X_train_shape[1], X_train_shape[2])),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(filters=128, kernel_size=5),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Activation('relu'),

        GlobalMaxPool1D(),

        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(y_train_shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def filter_task2(df):
    filtered_df = df[df['activity_label'].isin([0, 1, 2])]
    return filtered_df

training_folder = r'E:\Pdiot\training_breathingOnly'
eval_folder = r'E:\Pdiot\evaluate_breathingOnly'

train_data = sorted([file for file in os.listdir(training_folder) if file.endswith('.csv')])
eval_data = sorted([file for file in os.listdir(eval_folder) if file.endswith('.csv')])

assert len(train_data) == len(eval_data), "Directories do not have the same number of CSV files"

test_accuracies = []
test_accuracy_dict = dict()
mapping_matrix = np.full((11, 3), -1)  # Initialize with an invalid class label, e.g., -1
# Now fill in the matrix with the correct mappings
mapping_matrix[2, 0] = 3
mapping_matrix[2, 1] = 8
mapping_matrix[2, 2] = 13
mapping_matrix[3, 0] = 1
mapping_matrix[3, 1] = 6
mapping_matrix[3, 2] = 11
mapping_matrix[4, 0] = 4
mapping_matrix[4, 1] = 9
mapping_matrix[4, 2] = 14
mapping_matrix[5, 0] = 2
mapping_matrix[5, 1] = 7
mapping_matrix[5, 2] = 12
mapping_matrix[10, 0] = 0  # sitting/standing + breathing normally
mapping_matrix[10, 1] = 5  # sitting/standing + coughing
mapping_matrix[10, 2] = 10  # sitting/standing + hyperventilating


start_time = time.time()
for file1, file2 in zip(train_data, eval_data):
    path1 = os.path.join(training_folder, file1)
    path2 = os.path.join(eval_folder, file2)

    train_file = path1.split('_')[-1].lstrip()
    test_file = path2.split('_')[-1].lstrip()
    print(f"Left out: {train_file}")
    print(f"Test on: {test_file}")
    assert train_file == test_file, "Not same subjects"
    
    data_train_temp = pd.read_csv(path1)
    data_eval_temp = pd.read_csv(path2)

    # columns_to_drop = ['activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
    columns_to_drop = ['subject', 'activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
    data_train_temp = data_train_temp.drop(columns=columns_to_drop)
    data_eval_temp = data_eval_temp.drop(columns=columns_to_drop)
    
    data_train = filter_task2(data_train_temp)
    data_eval = filter_task2(data_eval_temp)

    window_size = 125  # 4 second window
    overlap_size = 45 # 50% overlap

    segments1, labels1 = create_segments_and_labels(data_train, window_size, overlap_size, "physical_label", "activity_label")
    test_segment1, test_labels1 = create_segments_and_labels(data_eval, window_size, overlap_size, "physical_label", "activity_label")

    segments2, labels2 = create_segments_and_labels(data_train, window_size, overlap_size, "activity_label", "physical_label")
    test_segment2, test_labels2 = create_segments_and_labels(data_eval, window_size, overlap_size,"activity_label", "physical_label")


    scaler1 = StandardScaler() # or MinMaxScaler() or StandardScaler() or RobustScaler() or QuantileTransformer(n_quantiles=20, random_state=0)
    segments1 = scaler1.fit_transform(segments1.reshape(-1, segments1.shape[-1])).reshape(segments1.shape)
    test_segment1 = scaler1.transform(test_segment1.reshape(-1, test_segment1.shape[-1])).reshape(test_segment1.shape)

    scaler2 = StandardScaler() 
    segments2 = scaler2.fit_transform(segments2.reshape(-1, segments2.shape[-1])).reshape(segments2.shape)
    test_segment2 = scaler2.transform(test_segment2.reshape(-1, test_segment2.shape[-1])).reshape(test_segment2.shape)

    # Reshape the segments for the CNN
    X_1 = np.array(segments1)  # Assuming segments is a list of 2D arrays (time steps, features)
    y_1 = to_categorical(labels1)  # One-hot encode the labels

    X_test_1 = np.array(test_segment1)
    y_test_1 = to_categorical(test_labels1)
    
    # Split the data into training and testing sets
    X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_1, y_1, test_size=0.3, random_state=42, stratify=y_1)

    # Reshape the segments for the CNN
    X_2 = np.array(segments2)  # Assuming segments is a list of 2D arrays (time steps, features)
    y_2 = to_categorical(labels2)  # One-hot encode the labels

    X_test_2 = np.array(test_segment2)
    y_test_2 = to_categorical(test_labels2)
    # Split the data into training and testing sets
    
    X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_2, y_2, test_size=0.3, random_state=42, stratify=y_2)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    
    original_length = max(X_test_1.shape[0], X_test_2.shape[0])
    
    model_1 = train_model_1(X_train_1.shape, y_train_1.shape)
    print("---------------------- Training model level 1 -----------------------")
    model_1.fit(X_train_1, y_train_1, epochs=10, batch_size=128, validation_data=(X_val_1, y_val_1), shuffle=True, callbacks=[early_stopping])

    model_2 = train_model_2(X_train_2.shape, y_train_2.shape)
    # Train the model with early stopping
    print("---------------------- Training model level 2 -----------------------")
    model_2.fit(X_train_2, y_train_2, epochs=10, batch_size=128, validation_data=(X_val_2, y_val_2), shuffle=True, callbacks=[early_stopping])

    # Evaluate the model

    model1_preds = model_1.predict(X_test_1)
    model2_preds = model_2.predict(X_test_2)  

    # Convert predictions to class labels
    model1_labels = np.argmax(model1_preds, axis=1)
    model2_labels = np.argmax(model2_preds, axis=1)

    # Convert true labels from one-hot encoding to class labels
    true_labels_1 = np.argmax(y_test_1, axis=1)
    true_labels_2 = np.argmax(y_test_2, axis=1)

    combined_predictions = []
    correct_labels = []
    incorrect_pred = 0 
    corr_pred = 0

    for i in range(len(y_test_1)):
        # Get the combined class from the mapping matrix
        combined_class = mapping_matrix[model1_labels[i], model2_labels[i]]

        pred = (model1_labels[i], model2_labels[i])
        correct = (true_labels_1[i], true_labels_2[i])

        correct_label = mapping_matrix[true_labels_1[i], true_labels_2[i]]
        correct_labels.append(correct_label)

        if pred != correct:
            incorrect_pred += 1
        else:
            corr_pred += 1
            
        if combined_class == -1:
            print(f"Invalid class combination: Model 1 class {model1_labels[i]}, Model 2 class {model2_labels[i]}")
        else:
            combined_predictions.append(combined_class)

    # Determine where both sets of predictions are correct
    correct_predictions = np.logical_and(model1_labels == true_labels_1, model2_labels == true_labels_2)

    # Use the mapping matrix to get the combined true labels
    y_true_combined = [mapping_matrix[l1][l2] for l1, l2 in zip(true_labels_1, true_labels_2)]

    # Calculate combined accuracy
    combined_accuracy = corr_pred/(corr_pred+incorrect_pred)
    print(f"Combined Accuracy: {combined_accuracy * 100:.2f}%")

    accuracy = combined_accuracy
    
    # Define all possible class labels
    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


    # Generate classification report

    # Generate the classification report
    report = classification_report(
        y_true_combined,
        combined_predictions,
        labels=all_labels,  # Explicitly provide all class labels
        target_names=[
            'class 0: sitting/standing + breathing normally',
            'class 1: lying down on your left side + breathing normally',
            'class 2: lying down on your right side + breathing normally',
            'class 3: lying down on your back + breathing normally',
            'class 4: lying down on your stomach + breathing normally',
            'class 5: sitting/standing + coughing',
            'class 6: lying down on your left side + coughing',
            'class 7: lying down on your right side + coughing',
            'class 8: lying down on your back + coughing',
            'class 9: lying down on your stomach + coughing',
            'class 10: sitting/standing + hyperventilating',
            'class 11: lying down on your left side + hyperventilating',
            'class 12: lying down on your right side + hyperventilating',
            'class 13: lying down on your back + hyperventilating',
            'class 14: lying down on your stomach + hyperventilating'
        ],
        zero_division=0  # Prevent division by zero errors if a class has no samples
    )

    print(report)

    print(f'Accuracy: {accuracy*100}%')
    test_accuracies.append(float(accuracy))
    print()
    print("---"*18)
    print(f"Average accuracy so far: {sum(test_accuracies) / len(test_accuracies)}")
    print("---"*18)
    print()
    # break
    del model_1
    del model_2

end_time = time.time()
average_accuracy = sum(test_accuracies) / len(test_accuracies)
std_deviation = np.std(test_accuracies)

print(f"Training time: {end_time - start_time}")
print(f"Average Test Accuracy: {average_accuracy * 100:.2f}%")
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"minimum = {min(test_accuracies)}")
