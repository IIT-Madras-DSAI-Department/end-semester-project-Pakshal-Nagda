from algorithms import SoftmaxRegression, DecisionTree, read_data
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import time

X_train, y_train, X_val, y_val = read_data()
train_acc = []
val_acc = []
train_f1 = []
val_f1 = []

for i in range(4, 14, 3):
    model = DecisionTree(max_depth=i, min_samples_split=10)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f'Model fit in {end - start:.3f} seconds')

    #y_pred = model.predict(X_train)
    #print(classification_report(y_train, y_pred))
    #ConfusionMatrixDisplay.from_predictions(y_train, y_pred)

    #y_pred = model.predict(X_val)
    #print(classification_report(y_val, y_pred))
    #ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    y_pred = model.predict(X_train)
    train_acc.append(accuracy_score(y_train, y_pred))
    train_f1.append(f1_score(y_train, y_pred, average='macro'))
    y_pred = model.predict(X_val)
    val_acc.append(accuracy_score(y_val, y_pred))
    val_f1.append(f1_score(y_val, y_pred, average='macro'))

plt.plot(range(4, 14, 3), train_acc, label='Train accuracy')
plt.plot(range(4, 14, 3), val_acc, label='Validation accuracy')
plt.plot(range(4, 14, 3), train_f1, label='Train F1 score')
plt.plot(range(4, 14, 3), val_f1, label='Validation F1 score')
plt.title('Decision Tree Performance')
plt.xlabel('max_depth')
plt.ylabel('Metric')
plt.legend()
plt.show()
