import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Config:    
    DATA_DIR = r'C:\Users\mutha\Desktop\Crime_Detection\data'
    OUTPUT_DIR = r'C:\Users\mutha\Desktop\Crime_Detection\models'    
    
    INPUT_SHAPE = (30, 64, 64, 3)  
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    AUGMENTATION = True

def load_and_preprocess_data(data_dir):
    X = []
    y = []
    crime_dir = os.path.join(data_dir, 'crime')
    for video_file in os.listdir(crime_dir):
        video_path = os.path.join(crime_dir, video_file)
        frames = load_video_frames(video_path)
        if len(frames) == 30:  
            X.append(frames)
            y.append(1)  
    non_crime_dir = os.path.join(data_dir, 'non_crime')
    for video_file in os.listdir(non_crime_dir):
        video_path = os.path.join(non_crime_dir, video_file)
        frames = load_video_frames(video_path)
        if len(frames) == 30:  
            X.append(frames)
            y.append(0)  
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def load_video_frames(video_path, target_size=(64, 64), max_frames=30):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break 
        frame = cv2.resize(frame, target_size)
        frame = frame.astype("float") / 255.0
        frames.append(frame)
    
    cap.release()
    if len(frames) < max_frames:
        frames.extend([frames[-1]] * (max_frames - len(frames)))
    
    return frames[:max_frames]

def create_3d_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
                      input_shape=input_shape, padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  
    ])
    
    return model


def train_model(X, y, config):
   
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42)
    model = create_3d_cnn_model(config.INPUT_SHAPE, config.NUM_CLASSES)
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(config.OUTPUT_DIR, 'crime_detection_model.h5'),
        save_best_only=True,
        monitor='val_accuracy'
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint])
    return model, history
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    print("Classification Report:")
    print(classification_report(y_val, y_pred_classes, 
                                target_names=['Non-Crime', 'Crime']))

def main():
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(Config.DATA_DIR)
    print("Training model...")
    model, history = train_model(X, y, Config)
    print("Evaluating model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.VALIDATION_SPLIT, random_state=42)
    evaluate_model(model, X_val, y_val)
if __name__ == '__main__':
    main()