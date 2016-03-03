import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from util import write_predictions

# Read in train and test as Pandas DataFrames
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# format training data
Y_train = df_train.clazz.values
df_train = df_train.drop(['Id'], axis=1)
df_train = df_train.drop(['clazz'], axis=1)
print df_train.head()
X_train = df_train.values

# format test data
test_ids = df_test.Id.values
df_test = df_test.drop(['Id'], axis=1)
X_test = df_test.values

print "Train features:", X_train.shape
print "Train class:", Y_train.shape
print "Test features:", X_test.shape


# RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)
write_predictions(RF_pred, test_ids, 'predicted_RF01.csv')

# # AdaBoostClassifier
# AB = AdaBoostClassifier()
# AB.fit(X_train, Y_train)
# AB_pred = AB.predict(X_test)
# write_predictions(AB_pred, test_ids, 'predicted_AB.csv')

# # GaussianNB
# NB = GaussianNB()
# NB.fit(X_train, Y_train)
# NB_pred = NB.predict(X_test)
#
# write_predictions(NB_pred, test_ids, 'predicted_NB.csv')

# # DecisionTreeClassifier
# DT = DecisionTreeClassifier()
# DT.fit(X_train, Y_train)
# DT_pred = DT.predict(X_test)
#
# write_predictions(DT_pred, test_ids, 'predicted_DT.csv')

# # DecisionTreeClassifier
# SVM = SVC(gamma=2, C=1)
# SVM.fit(X_train, Y_train)
# SVM_pred = SVM.predict(X_test)
#
# write_predictions(SVM_pred, test_ids, 'predicted_SVM.csv')

# # RandomForestClassifier
# KN = KNeighborsClassifier()
# KN.fit(X_train, Y_train)
# KN_pred = KN.predict(X_test)
#
# write_predictions(KN_pred, test_ids, 'predicted_KN.csv')
