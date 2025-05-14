#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import time
import os
import numpy as np
import joblib
import sounddevice as sd
from joblib import Parallel, delayed
import gc  # For garbage collection to free memory


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk(r"C:\Users\shifa\Desktop\archive\Converted_Separately"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


pip install imblearn


# In[4]:


import torch
print(torch.cuda.is_available())


# In[5]:


# Directories for Screaming and Not Screaming datasets
scream_dir = r"C:\Users\shifa\Desktop\archive\Converted_Separately\scream"
non_scream_dir = r"C:\Users\shifa\Desktop\archive\Converted_Separately\non_scream"
screaming_dir = r"C:\Users\shifa\Desktop\archive\Converted_Separately\screaming"
not_screaming_dir = r"C:\Users\shifa\Desktop\archive\Converted_Separately\notscreaming"

RATE = 44100  # Sample rate


# In[6]:


def extract_single_audio_features(y):
    """Extracts various audio features including MFCCs, spectral contrast, chroma features, and more."""
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=RATE)
    
    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=RATE)
    
    # Extract Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)
    
    # Extract Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=RATE)
    
    # Extract Root Mean Square Energy (RMSE)
    rmse = librosa.feature.rms(y=y)
    
    # Extract Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=RATE)
    
    # Extract Tempo (Beats Per Minute)
    onset_env = librosa.onset.onset_strength(y=y, sr=RATE)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=RATE)
    
    # Concatenate all the features into a single feature vector
    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),             # MFCCs
         np.mean(spectral_contrast, axis=1), # Spectral Contrast
        np.mean(chroma, axis=1),            # Chroma
        np.mean(zero_crossing_rate),        # Zero Crossing Rate
        np.mean(spectral_centroid),         # Spectral Centroid
        np.mean(spectral_bandwidth),        # Spectral Bandwidth
        np.mean(rmse),                      # RMSE
        np.mean(spectral_rolloff),          # Spectral Roll-off
        tempo                               # Tempo (BPM)
    ])
    
    return feature_vector


# In[7]:


def load_and_process_audio_files(directory):
    """Load all audio files from a directory, process and extract features."""
    audio_features = []
    
    for root, _, files in os.walk(directory):
        for file in files[:20]:
            file_path = os.path.join(root, file)
            y, _ = librosa.load(file_path, sr=RATE)
            features = extract_single_audio_features(y)
            audio_features.append(features)
    
    return np.array(audio_features)


# In[8]:


# Step 1: Load and process scream audio files
print("Loading and processing scream audio files...")
X_screams_1 = load_and_process_audio_files(screaming_dir)
print("1/2")
X_screams_2 = load_and_process_audio_files(scream_dir)
print("2/2")
    
# Combine scream datasets from both directories
X_screams = np.vstack([X_screams_1, X_screams_2])
print("Scream audio files processed and features extracted.")

# Step 2: Load and process non-scream audio files
print("Loading and processing non-scream audio files...")
X_non_screams_1 = load_and_process_audio_files(not_screaming_dir)
print("1/2")
X_non_screams_2 = load_and_process_audio_files(non_scream_dir)
print("2/2")
    
# Combine non-scream datasets from both directories
X_non_screams = np.vstack([X_non_screams_1, X_non_screams_2])
print("Non-scream audio files processed and features extracted.")
    
# Now you can proceed with model training using X_screams and X_non_screams
print(f"Scream features shape: {X_screams.shape}")
print(f"Non-scream features shape: {X_non_screams.shape}")


# In[9]:


# Step 3: Convert features to arrays and combine
X = np.vstack([X_screams, X_non_screams])
y_screams = np.ones(len(X_screams))
y_non_screams = np.zeros(len(X_non_screams))
y = np.hstack([y_screams, y_non_screams])
print("3. step completed...")


# In[10]:


print(X.shape)
print(y.shape)


# In[11]:


# Step 4: Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("4. step completed...")


# In[12]:


# Step 7: Handle class imbalance (optional)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("7. step completed...")


# In[13]:


# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("5. step completed...")


# In[14]:


print("RandomForestClassifier")

classifier = RandomForestClassifier()
    
# Fit the model and make predictions
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
    
# Evaluation
print(f"Accuracy :{round(accuracy_score(y_test, y_pred)*100, 2)}")
print(f"Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
print(f"Classification Report :\n", classification_report(y_test, y_pred))
print("...................")


# In[15]:


import joblib

# Example: save model to Desktop
joblib.dump(classifier, r"C:\Users\shifa\Desktop\archive\Converted_Separately\scream_detector_model.pkl")
joblib.dump(scaler, r"C:\Users\shifa\Desktop\archive\Converted_Separately\scaler.pkl")

REAL TIME SCREAM DETECTION
# In[16]:


# Constants
RATE = 44100  # Sample rate
CHUNK_DURATION = 1.0  # Duration of each chunk in seconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION)  # Number of samples per chunk


# In[17]:


# Load the pre-trained model and scaler
model = joblib.load(r"C:\Users\shifa\Desktop\archive\Converted_Separately\scream_detector_model.pkl")  
scaler = joblib.load(r"C:\Users\shifa\Desktop\archive\Converted_Separately\scaler.pkl")  


# In[18]:


"""# Define feature extraction function (same as used during training)
def extract_single_audio_features(y):
    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=RATE)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=RATE)
    rmse = librosa.feature.rms(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=RATE)
    
    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(rmse),
        np.mean(spectral_rolloff)
    ])
    return feature_vector
    """
def extract_single_audio_features(y):
    """Extract features - MUST MATCH TRAINING EXACTLY"""
    # Audio features
    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=RATE)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=RATE)
    rmse = librosa.feature.rms(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=RATE)
    
    # Tempo calculation (CRUCIAL - this was in training)
    onset_env = librosa.onset.onset_strength(y=y, sr=RATE)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=RATE)
    
    # MUST MATCH TRAINING ORDER EXACTLY
    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),             # 13 MFCCs
        np.mean(spectral_contrast, axis=1), # 7 spectral contrast
        np.mean(chroma, axis=1),            # 12 chroma
        np.mean(zero_crossing_rate),        # 1
        np.mean(spectral_centroid),         # 1
        np.mean(spectral_bandwidth),        # 1
        np.mean(rmse),                      # 1
        np.mean(spectral_rolloff),          # 1
        tempo                               # 1 (TOTAL: 13+7+12+1+1+1+1+1+1 = 38)
    ])
    
    return feature_vector


# In[19]:


"""# Callback function to process each audio chunk
def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    # Reshape and process the audio chunk
    audio_chunk = indata[:, 0]  # Use the first channel
    features = extract_single_audio_features(audio_chunk)

    # Reshape for scaling and scaling the features
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(scaled_features)

    if prediction == 1:  # 1 indicates scream
        print("Scream detected!")"""
def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    try:
        audio_chunk = indata[:, 0]
        features = extract_single_audio_features(audio_chunk)
        print("Extracted features count:", len(features))  # Should print 38
        
        features = features.reshape(1, -1)
        print("Feature shape before scaling:", features.shape)  # Should be (1, 38)
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        if prediction == 1:
            print("SCREAM DETECTED!")
    except Exception as e:
        print(f"Processing error: {str(e)}")


# In[20]:


# Start the microphone input stream
with sd.InputStream(callback=callback, channels=1, samplerate=RATE, blocksize=CHUNK_SIZE):
    print("Listening for screams...")
    while True:
        sd.sleep(int(CHUNK_DURATION * 1000))  # Wait for the chunk duration


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Training and evaluating RandomForestClassifier...")

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation
print(f"Accuracy : {round(accuracy_score(y_test, y_pred) * 100, 2)}")
print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report :\n{classification_report(y_test, y_pred)}")

# Save model and scaler
joblib.dump(classifier, "Converted_Separately/scream_detector_model.pkl")
joblib.dump(scaler, "Converted_Separately/scaler.pkl")
print("Model and scaler saved.")


# In[ ]:




