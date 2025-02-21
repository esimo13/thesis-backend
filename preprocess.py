import numpy as np
import joblib
from scipy.signal import stft, detrend, butter, filtfilt, iirnotch

# Load pre-trained scaler
scaler = joblib.load("scaler.pkl")

# Bandpass filter
def bandpass_filter(data, fs=1024, lowcut=0.5, highcut=50, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return np.array([filtfilt(b, a, channel) for channel in data])

# Notch filter (50Hz)
def notch_filter(data, fs=1024, freq=50, quality=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality)
    return np.array([filtfilt(b, a, channel) for channel in data])

# Detrending
def detrend_data(eeg_data):
    return np.array([detrend(channel, type='linear') for channel in eeg_data])

# Z-score Normalization
def normalize_data(eeg_data):
    return np.array([(channel - np.mean(channel)) / np.std(channel) for channel in eeg_data])

# Apply STFT
def apply_stft(channel_data, fs=1024):
    f, t, Zxx = stft(channel_data, fs=fs, window='hann', nperseg=2, noverlap=1)
    return np.abs(Zxx).flatten()[:1025]  # Ensure fixed size

# Main function to process EEG .txt file
def preprocess_input_txt(file_path, fs=1024):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    data = bandpass_filter(data, fs)
    data = notch_filter(data, fs)
    data = detrend_data(data)
    data = normalize_data(data)

    # Extract STFT features
    stft_result = np.array([apply_stft(channel, fs) for channel in data])
    input_features = stft_result.flatten()[:1025]  

    # Standardize features
    input_features = scaler.transform(input_features.reshape(1, -1))

    return input_features