from music_dataset import MusicDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

# -Desc.:
#   This function will find the best fit paramaters for a Gradient Boosted
# Tree Classifier given preprocessed dataset. The model will be trained using
# grid search w/ cross validation. Overall accuracy and individual class
# metrics will be printed out at the end.
# -Input:
#   X - predictor data, y - target data corresponding to X
def gradient_boosted(X, y):
    # Set possible parameters for model tuning
    parameters = {
        'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],
        'max_depth':np.linspace(1, 32, 32, endpoint=True)
    }
    # Split 20% testing, 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gbtc = GradientBoostingClassifier()
    gbtc_cv = GridSearchCV(gbtc, parameters)
    gbtc_cv.fit(X_train, y_train)
    y_pred = gbtc_cv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matricies = multilabel_confusion_matrix(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    best_params = gbtc_cv.best_params

    # Print out Gradient Boosted Tree report
    print('-Gradient Boosted Tree Report-')
    print('\tGradeint Boosted Tree overall accuracy: ', accuracy)
    print('\tBest parameters:')
    print('\t\tBest Learning Rate: ', best_params['learning_rate'])
    print('\t\tBest N Estimators: ', best_params['n_estimators'])
    print('\t\tBest Max Depth: ', best_params['max_depth'])
    print('\tMulticlass metrics:')

    # Print out individual class metrics
    for confusion in confusion_matricies:
        tp = confusion[0][0]
        fp = confusion[1][0]
        fn = confusion[0][1]
        tn = confusion[1][1]

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)

        print('\t\t<label> accuracy: ', accuracy)
        print('\t\t<label> recall: ', recall)
        print('\t\t<label> specificity: ', specificity)
        print('\t\t<label> precision: ', precision)

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
    gtzan_dataset = MusicDataset('../data/gtzan/genres_original')
    gtzan_dataset.load_music()


if __name__ == '__main__':
    main()
