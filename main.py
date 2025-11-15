from algorithms import (
    SoftmaxRegression,
    DecisionTree,
    RandomForest,
    SVM,
    Perceptron,
    OneVsAll,
    OneVsOne,
    PCA,
    VotingEnsembler,
    StackingEnsembler,
    read_data
)
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import time

X_train, y_train, X_val, y_val = read_data()

model = VotingEnsembler(
    [
        SoftmaxRegression(random_state=108),
        RandomForest(40, 6),
        OneVsOne(Perceptron, 150),
        OneVsAll(Perceptron, 500),
        OneVsAll(SVM)
    ]
)

start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f'Model fit in {end - start:.3f} seconds')

y_pred = model.predict(X_train)
print(f'Train Accuracy: {accuracy_score(y_train, y_pred):.3f} | F1 Score: {f1_score(y_train, y_pred, average="macro"):.3f}')
ConfusionMatrixDisplay.from_predictions(y_train, y_pred)

y_pred = model.predict(X_val)
print(f'Val Accuracy: {accuracy_score(y_val, y_pred):.3f} | F1 Score: {f1_score(y_val, y_pred, average="macro"):.3f}\n')
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)

plt.show()

# Test Data
df = pd.read_csv('MNIST_validation.csv')
X_test = df.drop(columns=['label', 'even']).values / 256
y_test = df['label'].values
y_pred = model.predict(X_test)
print('==============================')
print('Accuracy on Test Data:', accuracy_score(y_test, y_pred))
print('F1 Score on Test Data:', f1_score(y_test, y_pred))
