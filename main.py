from algorithms import (
    SoftmaxRegression,
    DecisionTree,
    RandomForest,
    SVM,
    Perceptron,
    OneVsAll,
    read_data
)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time

X_train, y_train, X_val, y_val = read_data()

model = OneVsAll(Perceptron, epochs=100)

start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f'Model fit in {end - start:.3f} seconds')

y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))
ConfusionMatrixDisplay.from_predictions(y_train, y_pred)

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)

plt.show()
