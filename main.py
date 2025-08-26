from operator import not_
from random import sample
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

# Read the credit card fraud dataset
df = pd.read_csv('creditcard.csv')

new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
time = new_df['Time']
new_df['Time'] = (time - time.min()) / (time.max() - time.min())

new_df = new_df.sample(frac=1, random_state=1)

train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]
#print(train['Class'].value_counts(), test['Class'].value_counts(), val['Class'].value_counts())

train_np, test_np, val_np = train.to_numpy(), test.to_numpy(), val.to_numpy()

# print(train_np.shape, test_np.shape, val_np.shape)

x_train, y_train = train_np[:, :-1], train_np[:, -1]
x_test, y_test = test_np[:, :-1], test_np[:, -1]
x_val, y_val = val_np[:, :-1], val_np[:, -1]

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)


logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
logistic_model.score(x_train, y_train)       #99.9% accuracy

# ?print(classification_report(y_val, logistic_model.predict(x_val), target_names=['Not Fraud', 'Fraud']))


#                   Predicted Fraud (+)                 Predicted Not Fraud(-)
# Fraud (+)         True Positive                       False Negative
# Not Fraud(-)      False Positive                      True Negative

#              precision    recall  f1-score   support

#    Not Fraud       1.00      1.00      1.00     22771
#        Fraud       0.83      0.56      0.67        36

#     accuracy                           1.00 (skewed)     22807
#    macro avg       0.92      0.78      0.83     22807
# weighted avg       1.00      1.00      1.00     22807


# can't just look at accuracy, and have to look at precision and recall too

shallow_nn = Sequential()
shallow_nn.add(InputLayer((x_train.shape[1],)))
shallow_nn.add(Dense(2, 'relu'))
shallow_nn.add(BatchNormalization())
shallow_nn.add(Dense(1, 'sigmoid'))

checkpoint = ModelCheckpoint('shallow_nn.keras', save_best_only=True)
shallow_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# print(shallow_nn.summary())


#  shallow_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=checkpoint)

def neural_net_predictions(model, x):
    return (shallow_nn.predict(x).flatten() > 0.5).astype(int)
# print(neural_net_predictions(shallow_nn, x_val))

# print(classification_report(y_val, neural_net_predictions(shallow_nn, x_val), target_names=['Not Fraud', 'Fraud']))


# ----------------------------- RANDOM FOREST MODEL -----------------------------

# rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
# rf.fit(x_train, y_train)
# print(classification_report(y_val, rf.predict(x_val), target_names=['Not Fraud', 'Fraud']))


# ----------------------------- Gradient Boosting Model -----------------------------

# gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
# gbc.fit(x_train, y_train)
# print(classification_report(y_val, gbc.predict(x_val), target_names=['Not Fraud', 'Fraud']))

# ----------------------------- Support Vector Machine Model -----------------------------

# svc = LinearSVC()
# # svc = LinearSVC(class_weight='balanced') - this is too aggressive and gives me 0.07 precision Fraud
# svc.fit(x_train, y_train)
# print(classification_report(y_val, svc.predict(x_val), target_names=['Not Fraud', 'Fraud']))


# ----------------------------- STATS -----------------------------
#                           P         R         f-1 (balance between P & R)
# Logistic on val:          0.83      0.56      0.67      
# shallow_nn on val:        0.68      0.75      0.71            (THIS IS USUALLY BETTER)

# random for on val:        0.77      0.47      0.59
# gradient boos val:        0.67      0.67      0.67
# linaer svm on val:        0.90      0.74      0.80


not_frauds = new_df.query('Class == 0')
frauds = new_df.query('Class == 1')
# print(not_frauds['Class'].value_counts(), frauds['Class'].value_counts())

balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])
# print(balanced_df['Class'].value_counts())

balanced_df = balanced_df.sample(frac=1, random_state=1)

balanced_df_np = balanced_df.to_numpy()

x_train_b, y_train_b = balanced_df_np[:700, :-1],  balanced_df_np[:700, -1]         # uses first 700 of 982 vals
x_test_b, y_test_b = balanced_df_np[700: 842, :-1],  balanced_df_np[700:842, -1]    # uses 700-842 vals, 142 total
x_val_b, y_val_b = balanced_df_np[842:, :-1],  balanced_df_np[842:, -1]    # uses 700-842 vals, 142 total

print(x_train_b.shape, y_train_b.shape, x_test_b.shape, y_test_b.shape, x_val_b.shape, y_val_b.shape)
