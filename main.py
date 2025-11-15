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
import time

X_train, y_train, X_val, y_val = read_data()

train_acc = []
val_acc = []
train_f1 = []
val_f1 = []
n_components = [10, 50, 100, 200, 500, 784]
for n in n_components:
    pca = PCA(n)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    model = OneVsAll(Perceptron, 500)

    start = time.time()
    model.fit(X_train_pca, y_train)
    end = time.time()
    print(f'Model {n} fit in {end - start:.3f} seconds')

    y_pred = model.predict(X_train_pca)
    train_acc.append(accuracy_score(y_train, y_pred))
    train_f1.append(f1_score(y_train, y_pred, average='macro'))
    print(f'Train Accuracy: {train_acc[-1]:.3f} | F1 Score: {train_f1[-1]:.3f}')
    #ConfusionMatrixDisplay.from_predictions(y_train, y_pred)

    y_pred = model.predict(X_val_pca)
    val_acc.append(accuracy_score(y_val, y_pred))
    val_f1.append(f1_score(y_val, y_pred, average='macro'))
    print(f'Val Accuracy: {val_acc[-1]:.3f} | F1 Score: {val_f1[-1]:.3f}\n')
    #ConfusionMatrixDisplay.from_predictions(y_val, y_pred)

plt.plot(n_components, train_acc, 'o--', label='Training Accuracy')
plt.plot(n_components, val_acc, 'o-', label='Validation Accuracy')
plt.plot(n_components, train_f1, 's--', label='Training F1 Score')
plt.plot(n_components, val_f1, 's-', label='Validation F1 Score')
plt.title('Performance of Perceptron (OvA) with PCA')
plt.xlabel('PCA Components')
plt.ylabel('Metric')
plt.legend()
plt.show()
