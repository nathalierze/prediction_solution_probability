# %%
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=1077, activation="relu"))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


# load data
infile = open("../data_solution_probability_model.pkl", "rb")
df = pickle.load(infile)
infile.close()

# prepare features
feature_cols = list(df.columns)
feature_cols.remove("Erfolg")
X = df[feature_cols].astype(float)
y = df.Erfolg
y = y.astype("int")

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = build_model()

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    batch_size=128,
    verbose=0,
    validation_data=(X_test, y_test),
)

scores = model.evaluate(x=X_test, y=y_test, verbose=0)

# predict
yhat_probs = model.predict(X_test, verbose=0)
yhat_classes = (model.predict(X_test) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# save scores
accuracy = accuracy_score(y_test, yhat_classes)
precision = precision_score(y_test, yhat_classes)
recall = recall_score(y_test, yhat_classes)
f1 = f1_score(y_test, yhat_classes)
auc = roc_auc_score(y_test, yhat_probs)

# print scores
print("Accuracy: %.8f" % accuracy)
print("Precision: %.8f" % precision)
print("Recall: %.8f" % recall)
print("F1: %.8f" % f1)
print("AUC: %.8f" % auc)

# print confusion matrix
metrics.confusion_matrix(y_test, yhat_classes)

# save model as pickle dump
model.save("nn")
pickle.dump(X_train, open("X_train.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))
pickle.dump(yhat_probs, open("df_prob.pkl", "wb"))
