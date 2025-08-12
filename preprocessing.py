import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

def resampleData(train_dataset, test_dataset):
    
    df_1 = train_dataset[train_dataset[187] == 1]
    df_2 = train_dataset[train_dataset[187] == 2]
    df_3 = train_dataset[train_dataset[187] == 3]
    df_4 = train_dataset[train_dataset[187] == 4]

    df_1_upsample = resample(df_1, n_samples = 20000, replace = True, random_state = 123)
    df_2_upsample = resample(df_2, n_samples = 20000, replace = True, random_state = 123)
    df_3_upsample = resample(df_3, n_samples = 20000, replace = True, random_state = 123)
    df_4_upsample = resample(df_4, n_samples = 20000, replace = True, random_state = 123)
    df_0 = train_dataset[train_dataset[187]==0].sample(n =20000, random_state=123)
    train_dataset = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

    df_1_test = test_dataset[test_dataset[187] == 1]
    df_2_test = test_dataset[test_dataset[187] == 2]
    df_3_test = test_dataset[test_dataset[187] == 3]
    df_4_test = test_dataset[test_dataset[187] == 4]

    df_1_upsample_test = resample(df_1_test, n_samples = 5000, replace = True, random_state = 123)
    df_2_upsample_test = resample(df_2_test, n_samples = 5000, replace = True, random_state = 123)
    df_3_upsample_test = resample(df_3_test, n_samples = 5000, replace = True, random_state = 123)
    df_4_upsample_test = resample(df_4_test, n_samples = 5000, replace = True, random_state = 123)
    df_0_test = test_dataset[test_dataset[187]==0].sample(n =5000, random_state=123)
    test_dataset = pd.concat([df_0_test, df_1_upsample_test, df_2_upsample_test, df_3_upsample_test, df_4_upsample_test])

    return train_dataset, test_dataset

def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
  """
  order: filter order (5 — smooth transition between passed and suppressed frequencies).
  """
  nyquist_freq = 0.5 * fs
  normal_cutoff = cutoff_freq / nyquist_freq
  b, a = butter(order, normal_cutoff, btype='low', analog=False) # creates Butterworth filter coefficients.
  filtered_data = filtfilt(b, a, data) #  applies a forward and reverse filter — so as not to distort the signal phase (important in ECG!).
  
  return filtered_data

def cleanData(train_dataset, test_dataset):
    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values

    X_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values

    ecg_data = train_dataset
    ecg_signal_train = X_train
    ecg_signal_test = X_test
    class_labels = y_train
    sampling_rate = 125 # Sampling Frequency: 125Hz

    cutoff_frequency = 50 # Cutoff frequency — anything above 50 Hz will be suppressed (considered noise).
    train_filtered_ecg_signal = butter_lowpass_filter(ecg_signal_train, cutoff_frequency, sampling_rate)
    test_filtered_ecg_signal = butter_lowpass_filter(ecg_signal_test, cutoff_frequency, sampling_rate)

    X_train, X_valid, y_train, y_valid = train_test_split(train_filtered_ecg_signal, y_train, test_size=0.2, random_state=42)

    X_test = test_filtered_ecg_signal

    scaler = StandardScaler()

    X_train_normalized = scaler.fit_transform(X_train)
    X_valid_normalized = scaler.transform(X_valid)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_valid_normalized, X_test_normalized, y_train, y_valid, y_test, class_labels
