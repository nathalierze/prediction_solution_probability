# %%
import pandas as pd
import pickle
from sklearn import metrics 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from numpy import mean, absolute

# %%
infile = open('smallSampleSet.pkl','rb')
import_file = pickle.load(infile)
infile.close()

# %%
df = import_file.drop(columns=['UserID'])
feature_cols = list(df.columns)
feature_cols.remove('Erfolg')

# %%
X = df[feature_cols]
y = df.Erfolg
y= y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
k = 5
cv = KFold(n_splits=k, random_state=None)

clf = SVC(kernel='linear',gamma=1, probability=True)
clf = clf.fit(X_train,y_train)

scores_a = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
a = mean(absolute(scores_a))

scores_p = cross_val_score(clf, X_test, y_test, scoring='precision', cv=cv, n_jobs=-1)
p = mean(absolute(scores_p))

scores_r = cross_val_score(clf, X_test, y_test, scoring='recall', cv=cv, n_jobs=-1)
r = mean(absolute(scores_r))

scores_f1 = cross_val_score(clf, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1)
f1 = mean(absolute(scores_f1))

pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)


# %%
print("Accuracy: %.2f" %a)
print("Precision: %.2f" %p)
print("Recall: %.2f" %r)
print("F1: %.2f" %f1)


# %%
metrics.confusion_matrix(y_test, pred)

# save model as pickle dump
pickle.dump(clf, open('SVMmodel.pkl', 'wb'))
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))

# %%
#probs
t = probs[:,:1].tolist()
data_df = pd.DataFrame(t)
data_df

# %%
pickle.dump(data_df, open('df_prob.pkl', 'wb'))


