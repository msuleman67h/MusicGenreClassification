from glob import glob
from os.path import basename

import librosa.display
import numpy as np
from librosa import stft, amplitude_to_db
from librosa.feature import melspectrogram, mfcc
from matplotlib import pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm


class MusicDataset:
    def __init__(self, path, audio_sample_rate):
        self.path = path
        self.music_dataset = []
        self.audio_sample_rate = audio_sample_rate

    def load_music(self, is_sorted_by_genre: bool = True):
        if is_sorted_by_genre:
            sub_directories = glob(f"{self.path}/*")
            pbar = tqdm(sub_directories)
            for genre_subdir in pbar:
                pbar.set_description(f"Loading {basename(genre_subdir)}")
                music_files = glob(f"{genre_subdir}/*")
                for music_file in music_files:
                    try:
                        track_sample_rate, data = wavfile.read(music_file)
                        if track_sample_rate != self.audio_sample_rate:
                            print(f"Skipping track {music_file} with sample rate != {self.audio_sample_rate}")
                            continue
                        # normalize the music data
                        data = data / 2 ** (16 - 1)
                        self.music_dataset.append((data, basename(genre_subdir)))
                    except ValueError:
                        # Skipping one corrupt track jazz.00054.wav
                        pass

    def extract_stft_db_scale(self):
        for music in self.music_dataset:
            short_fourier_transform = stft(music[0], hop_length=256)
            music_scaled = np.abs(short_fourier_transform) ** 2
            amplitude_to_db(music_scaled, ref=np.max)

    # def extract_mel_spectrogram(self):
    #     for music in self.music_dataset:
    #         mel_spec = melspectrogram(music[0], sr=music[1])
    #         mel_spec_db = amplitude_to_db(mel_spec, ref=np.max)
    #         self.mel_spectrogram.append(mel_spec_db)
    #
    # def extract_mfcc(self):
    #     for music in self.music_dataset:
    #         self.mfccs.append(mfcc(music[0], n_mfcc=13, sr=music[1]))
    #         plt.figure(figsize=(25, 10))
    #         plt.figure(figsize=(25, 10))
    #         librosa.display.specshow(self.mfccs[0],
    #                                  x_axis="time",
    #                                  sr=music[1])
    #         plt.colorbar(format="%+2.f")
    #         plt.show()
    #         print("xx")
