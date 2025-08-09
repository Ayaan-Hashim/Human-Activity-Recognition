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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler

def create_segments_and_labels(df, window_size, step_size, target_col = "physical_label"):
    segments = []
    labels = []
    for start in range(0, len(df) - window_size, step_size):
        segment = df.iloc[start:start+window_size]
        label = mode(segment[target_col])[0]
        segment = segment.drop(columns = target_col)
        segments.append(segment)
        labels.append(label)
    
    return np.array(segments), np.array(labels)

def train_mopdel(X_train_shape,y_train_shape):
    model_architecture = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=3, input_shape=(X_train_shape[1], X_train_shape[2])),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(filters=128, kernel_size=3),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Activation('relu'),


        GlobalMaxPool1D(),

        Dense(32, activation='relu'),

        Dropout(0.3),
        Dense(y_train_shape[1], activation='softmax')
    ])

        # Compile the model
    model_architecture.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_architecture

def filter_only_normal(df):
    filtered_df = df[df['activity_label'].isin([0, -1])]
    return filtered_df


training_folder = r'E:\Pdiot\training_breathingOnly'
eval_folder = r'E:\Pdiot\evaluate_breathingOnly'

train_data = sorted([file for file in os.listdir(training_folder) if file.endswith('.csv')])
eval_data = sorted([file for file in os.listdir(eval_folder) if file.endswith('.csv')])

assert len(train_data) == len(eval_data), "Directories do not have the same number of CSV files"

test_accuracies = []
test_accuracy_dict = dict()

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
    columns_to_drop = ['subject', 'activity', 'timestamp']
    data_train_temp = data_train_temp.drop(columns=columns_to_drop)
    data_eval_temp = data_eval_temp.drop(columns=columns_to_drop)

    data_train = filter_only_normal(data_train_temp).drop(columns="activity_label")
    data_eval = filter_only_normal(data_eval_temp).drop(columns="activity_label")

    window_size = 75  # 3 second window
    overlap_size = 30  # 50% overlap 

    segments, labels = create_segments_and_labels(data_train, window_size, overlap_size)
    print(segments.shape)
    test_segment, test_labels = create_segments_and_labels(data_eval, window_size, overlap_size)

    # Apply Standard Scaler or Min-Max Scaler
    scaler = StandardScaler() # or MinMaxScaler() or StandardScaler() or RobustScaler() or QuantileTransformer(n_quantiles=20, random_state=0)
    segments = scaler.fit_transform(segments.reshape(-1, segments.shape[-1])).reshape(segments.shape)
    test_segment = scaler.transform(test_segment.reshape(-1, test_segment.shape[-1])).reshape(test_segment.shape)

    # Reshape the segments for the CNN
    X = np.array(segments)  # Assuming segments is a list of 2D arrays (time steps, features)
    y = to_categorical(labels)  # One-hot encode the labels

    X_test = np.array(test_segment)
    y_test = to_categorical(test_labels)
    # Split the data into training and testing sets
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = train_mopdel(X_train.shape, y_train.shape)

    # Define the EarlyStopping callback
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_val, y_val), shuffle=True, callbacks=[early_stopping])
    # model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val), shuffle=True)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy*100}%')
    test_accuracies.append(float(accuracy))
    print()
    print("---"*18)
    print(f"Average accuracy so far: {sum(test_accuracies) / len(test_accuracies)}")
    print("---"*18)
    print()
    
    del model

end_time = time.time()
average_accuracy = sum(test_accuracies) / len(test_accuracies)
std_deviation = np.std(test_accuracies)

print(f"Training time: {end_time - start_time}")
print(f"Average Test Accuracy: {average_accuracy * 100:.2f}%")
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"minimum = {min(test_accuracies)}")
