# Music Genre Classification  
  
The goal of this project is to identify the music genre such as pop, rock, hip-hop, metal, etc...  
  
## Dataset Description  
For dataset choice, we decided to use GTZAN Dataset[1], which is made up of ten different musical classes with 100 WAV files for each. The dataset was created by George Tzanetakis and was used in a well known paper in genre classification "Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002[2]. The files in the dataset were recorded from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. Each WAV file contains a 30 second clip of a song sampled at 22, 050 Hz with 16-bit mono audio in .WAV format.   
  
Additionally, we have the option of using the Free Music Archive (FMA) dataset found in the UCI Machine Learning Repository[3]. The structure of the dataset is similar to the GTZAN dataset, because of this we should be able to use it along with the GTZAN dataset for the training of the model. The smallest dataset in FMA has 8,000 tracks which are divided equally into 8 genres. The sampling rate of most of the tracks is 44,100 Hz with a bit rate of 320 kbit/s and is in stereo.

\[1\]: http://marsyas.info/downloads/datasets.html

\[2\]: Tzanetakis, George, and Perry Cook. "Musical genre classification of audio signals." IEEE Transactions on speech and audio processing 10.5 (2002): 293-302.

\[3\]: https://archive.ics.uci.edu/ml/datasets/FMA:+A+Dataset+For+Music+Analysis