from music_dataset import MusicDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.title("Spectrogram with decibel log", fontsize=24)
    plt.ylabel("Frequency (Hz)", fontsize=24)
    plt.xlabel("Time (s)", fontsize=24)
    plt.tight_layout()
    plt.show()

def main():
    gtzan_dataset = MusicDataset('../data/gtzan/genres_original', 22050)
    gtzan_dataset.load_music(is_sorted_by_genre=True)
    # gtzan_dataset.extract_stft_db_scale()
    # gtzan_dataset.extract_mel_spectrogram()
    # gtzan_dataset.extract_mfcc()
    print("ok")


if __name__ == '__main__':
    main()
