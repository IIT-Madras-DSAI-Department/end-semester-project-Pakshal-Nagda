from algorithms import SoftmaxRegression, read_data
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val = read_data()

model = SoftmaxRegression(epochs=150, random_state=108)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))
ConfusionMatrixDisplay.from_predictions(y_train, y_pred)

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)

plt.show()
