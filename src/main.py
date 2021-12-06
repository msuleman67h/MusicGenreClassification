from music_dataset import MusicDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

def k_nearest_neighbors(X, y):
    '''This function does the nearest neighbor stuff'''

    # Split the data into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

    # Set parameter grid to test over
    param_grid = {
        'n_neighbors': [1, 2, 3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }

    # Create KNN Classifier and perform grid search cross validation
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid)
    knn_cv.fit(X_train, y_train)
    y_pred = knn_cv.predict(X_test)
    accuracy = accuracy_score(X_test, y_pred)
    best_params = knn_cv.best_params_
    per_class_confusions = multilabel_confusion_matrix(y_test, y_pred)

    # Print out Gradient Boosted Tree report
    print('-K Nearest Neighbors Report-')
    print('\nKNN Overall Accuracy: ', accuracy)
    print('\tBest Parameters:')
    print('\t\tBest Number of Neighbors: ', best_params['n_neighbors'])
    print('\t\tBest Weight Type: ', best_params['weights'])
    print('\t\tBest Power Parameter (Minkowski): ', best_params['p'])
    print('\tMulticlass metrics:')

    # Print out individual class metrics
    for i, confusion in enumerate(per_class_confusions):
        tp = confusion[0][0]
        fp = confusion[1][0]
        fn = confusion[0][1]
        tn = confusion[1][1]

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)

        print(f'\t\t{i} accuracy: ', accuracy)
        print(f'\t\t{i} recall: ', recall)
        print(f'\t\t{i} specificity: ', specificity)
        print(f'\t\t{i} precision: ', precision)
    print()

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
