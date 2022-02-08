# Music Genre Classification 🎶

In this study, we conduct a multiclass music genre classification experiment by using signal processing techniques on music waveform data. We first calculate Mel-Frequency Cepstral Coefficients (MFCCs) to extract features from each sample of our data and use the mean value of the frequency bins as our final tabulated data. Finally, we test the preprocessed data with two non-linear multiclass classification models and analyze the performance: K-Nearest Neighbors (K-NN) and Gradient Boosted Forests (GBF). After using cross-validation, our best overall K-NN result is 49% accuracy, while our best GBF scored 66%.

# Table of Contents
1. [Dataset](#dataset)
2. [Data Preparation](#data-preparation)
3. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
4. [Model Development](#model-development)
5. [Conclusions](#conclusions)


## Dataset
For our dataset choice, we decided to use the GTZAN Dataset, which is made up of ten different musical classes (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock) with 100 tracks for each genre. The dataset was created by George Tzanetakis and was utilized in a well-known paper in genre classification "Musical Genre Classification of Audio Signals" by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002. The files in the dataset were recorded from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. Each audio file contains a 30-second clip of a song sampled at 22,050 Hz with 16-bit mono audio in .WAV format.

## Data Preparation
For reading the dataset, we used the SciPy library, which enabled us to store the waveform in the NumPy array. The WAV format stores the signal using Pulse-code modulation (PCM), obtained by sampling the amplitude signal at equal intervals. Therefore, the data we got was in the Amplitude time domain. Since the dataset is balanced, we didn't have to worry about the model skewing towards one genre. There was one corrupt track that we skipped over while reading the dataset. Lastly, we normalized the GTZAN dataset between 1 and -1 to make it easier for us to work with machine learning models.

Figure 1: Amplitude Time graph of one of the extracted music

## Preprocessing and Feature Engineering
Our main challenge working with time-series data was how to adapt it to work with traditional machine learning. Generally speaking, machine learning does not do well with time series and requires manual feature extraction. On the other hand, deep learning can do self feature extraction and time series and is currently an industry standard. To bring up machine learning to the same level as deep learning, we heavily relied on signal processing. Our end goal was to reduce the number of dimensions while extracting the principal features from the audio track.

Figure 2: Comparison of Machine Learning and Deep Learning

The first thing we did was to extract Mel-Spectrogram with the help of Short Fourier Transform (SFT). The Mel-Spectrogram was further used to derive the Mel-frequency cepstral coefficients (MFCCs) that are the frequency envelope of a signal. We ended up using all the MFCCs coefficients because we thought they were all critical for our task at hand.


## Model Development
Since the nature of the relationships between the target and the MFCCs is slightly complex, we decided to perform a grid search over several configurations of each model we tested. In this section, we’ll discuss why we chose certain parameters and show which performed best.

## Conclusions
In this experiment, we were able to demonstrate the performance of using MFCCs and traditional signal processing methods for machine learning algorithms. In the end, the resulting accuracies exceeded our expectations. They are significantly greater than random chance. All the resulting accuracies for individual classes performed greater than 80% while Gradient Boosted forest performed above 60%. Although our results show that the implementation of this algorithm may not be fit for production, we have been able to demonstrate empirical evidence that pursuing a signal processing to feature extraction may show a promising path. 
