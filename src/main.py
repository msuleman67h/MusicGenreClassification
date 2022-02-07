import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
import random

from my_lstm import MyLSTM

from music_dataset import MusicDataset


def plot_mfccs(y, sr, title):
    fig = plt.figure(figsize=(25, 10))
    librosa.display.specshow(y,
                             sr=sr,
                             x_axis="time")
    plt.colorbar(format="%+2.f")
    plt.title(title, fontsize=24)
    plt.ylabel("MFCCs Coefficients", fontsize=24)
    plt.xlabel("Time (s)", fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(left=0.07, right=1.0)
    plt.show()


def plot_mel_spectrogram(y, sr, hop_length):
    fig = plt.figure(figsize=(25, 10))
    librosa.display.specshow(y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis="log")
    plt.colorbar(format="%+2.f")
    plt.title("Mel-Spectrogram with decibel log", fontsize=24)
    plt.ylabel("Frequency (Hz)", fontsize=24)
    plt.xlabel("Time (s)", fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(left=0.07, right=1.0)
    plt.show()


def main():
    gtzan_dataset = MusicDataset('../data/genres_original')
    gtzan_dataset.load_music()
    track_no = random.randint(0, 999)
    track = gtzan_dataset.mfccs[track_no].cpu().detach().numpy()
    # plot_mel_spectrogram(track, sr=gtzan_dataset.audio_sample_rate, hop_length=256)
    # plot_mfccs(track, sr=gtzan_dataset.audio_sample_rate, title=f"MFCCs of track no {track_no}, genre {gtzan_dataset.target[track_no]}")
    print(gtzan_dataset.mfccs.shape)

    le = LabelEncoder()
    le.fit(gtzan_dataset.target)
    target = le.transform(gtzan_dataset.target)
    onehot_encoded_target = one_hot(torch.from_numpy(target)).to(device)
    onehot_encoded_target = onehot_encoded_target.double()

    full_data = list(zip(gtzan_dataset.mfccs, onehot_encoded_target))
    random.shuffle(full_data)

    training = full_data[:int(len(full_data) * 0.8)]
    testing = full_data[int(len(full_data) * 0.8):]

    my_lstm = MyLSTM(gtzan_dataset.mfccs.shape[2], gtzan_dataset.mfccs.shape[1], 2, 10).to(device)

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_lstm.parameters(), lr=0.001)
    num_epochs = 25
    for epoch in range(num_epochs):
        for index, data in enumerate(training):
            output = my_lstm(torch.unsqueeze(data[0], 0))
            loss = criterion(output, torch.unsqueeze(data[1], 0))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for index, data in enumerate(testing):
            output = my_lstm(torch.unsqueeze(data[0], 0))

            predicted = torch.argmax(output)
            true = torch.argmax(data[1])
            n_samples += 1
            if true == predicted:
                n_correct += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy: {acc} %')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
