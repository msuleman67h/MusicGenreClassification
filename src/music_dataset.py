from glob import glob
from os.path import basename

import numpy as np
import torchlibrosa
from librosa import amplitude_to_db
from librosa.feature import melspectrogram, mfcc
from scipy.io import wavfile
from tqdm import tqdm
import torch


class MusicDataset:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, path, audio_sample_rate: int = 22050):
        self.path = path
        self.audio_sample_rate = audio_sample_rate
        self.music_dataset = []
        self.target = []
        self.mel_spectrogram = None
        self.mfccs = []
        self.reduced_mfccs = []

    def load_music(self):
        """
        Loads each sound track from GTZAN data set to a list of numpy array. Also, stores their respective
        genres in a separate list.
        """
        min_track_length = 31.0
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
                    # Min-max normalizing the music data
                    data = data / 2 ** (16 - 1)
                    self.music_dataset.append(torch.from_numpy(data).float().to(MusicDataset.device))
                    track_length = data.shape[0] / self.audio_sample_rate
                    # keeping track of the smallest track, all the track will trim to this point.
                    if track_length < min_track_length:
                        min_track_length = track_length
                    self.target.append(basename(genre_subdir))
                except ValueError:
                    # Skipping over a corrupted track
                    pass
        print(f"The minimum track length is {min_track_length} seconds")
        # Making sure all the tracks are of same length.
        self.music_dataset = [track[:int(min_track_length * self.audio_sample_rate)] for track in self.music_dataset]
        self.music_dataset = torch.stack(self.music_dataset).to(MusicDataset.device)
        self.extract_mel_spectrogram()
        self.extract_mfccs()
        self.extract_reduced_mfccs()

    def extract_mel_spectrogram(self):
        """
        Extracts the Mel-Spectrogram for each track and stores them in instance variable
        """
        # Spectrogram
        spectrogram_extractor = torchlibrosa.Spectrogram(n_fft=1024, hop_length=256).to(MusicDataset.device)
        spectrogram_out = spectrogram_extractor.forward(self.music_dataset)

        logmel_extractor = torchlibrosa.LogmelFilterBank(sr=self.audio_sample_rate, n_fft=1024, n_mels=128).to(MusicDataset.device)
        self.mel_spectrogram = torch.squeeze(logmel_extractor.forward(spectrogram_out))

    def extract_mfccs(self):
        """
        Extracts the MFCCs for each track and stores them in instance variable
        """
        for music, log_mel_spec in tqdm(zip(self.music_dataset, self.mel_spectrogram), desc="Extracting MFCC's"):
            temp = mfcc(music, S=log_mel_spec, n_mfcc=39, sr=self.audio_sample_rate)
            self.mfccs.append(temp)

    def extract_reduced_mfccs(self):
        """
        Reduces the dimension of MFCCs and saves it in a separate instance variable
        """
        for mfcc in tqdm(self.mfccs, desc="Reducing MFCC's"):
            mean_var_coeff = []
            for mfcc_coeff in mfcc:
                mean_var_coeff.append(np.mean(mfcc_coeff))
                mean_var_coeff.append(np.var(mfcc_coeff))
            self.reduced_mfccs.append(mean_var_coeff)
