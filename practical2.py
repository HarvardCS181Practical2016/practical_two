import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from util import write_predictions
from feature_extraction import categorization_accuracy

# Read in train and test as Pandas DataFrames
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# format training data
Y_train = df_train.clazz.values
df_train = df_train.drop(['Id'], axis=1)
df_train = df_train.drop(['clazz'], axis=1)
# print df_train.head()
X_train = df_train.values

# format test data
test_ids = df_test.Id.values
df_test = df_test.drop(['Id'], axis=1)
X_test = df_test.values

print "Train features:", X_train.shape
print "Train class:", Y_train.shape
print "Test features:", X_test.shape


# RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, max_features='log2')
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)
write_predictions(RF_pred, test_ids, 'predicted_RF.csv')
# print 'RandomForestClassifier', categorization_accuracy('predicted_RF01.csv', 'actual_small.csv')

# 0.891444342226 - n_estimators=100, max_features='None'
# 0.894204231831 - n_estimators=50, max_features='log2'
# 0.897884084637 - n_estimators=100, max_features='log2'
# 0.896964121435 - n_estimators=125, max_features='log2'
# 0.896044158234 - n_estimators=115, max_features='log2'

# # QuadraticDiscriminantAnalysis
# QD = QuadraticDiscriminantAnalysis()
# QD.fit(X_train, Y_train)
# QD_pred = QD.predict(X_test)
# write_predictions(QD_pred, test_ids, 'predicted_QD.csv')
# print 'QuadraticDiscriminantAnalysis', categorization_accuracy('predicted_QD.csv', 'actual_small.csv')
#
#
# # AdaBoostClassifier
# AB = AdaBoostClassifier()
# AB.fit(X_train, Y_train)
# AB_pred = AB.predict(X_test)
# write_predictions(AB_pred, test_ids, 'predicted_AB.csv')
# print 'AdaBoostClassifier', categorization_accuracy('predicted_AB.csv', 'actual_small.csv')
#
# # GaussianNB
# NB = GaussianNB()
# NB.fit(X_train, Y_train)
# NB_pred = NB.predict(X_test)
# write_predictions(NB_pred, test_ids, 'predicted_NB.csv')
# print 'GaussianNB', categorization_accuracy('predicted_NB.csv', 'actual_small.csv')
#
# # DecisionTreeClassifier
# DT = DecisionTreeClassifier()
# DT.fit(X_train, Y_train)
# DT_pred = DT.predict(X_test)
# write_predictions(DT_pred, test_ids, 'predicted_DT.csv')
# print 'DecisionTreeClassifier', categorization_accuracy('predicted_DT.csv', 'actual_small.csv')
#
# # SVC
# SVM = SVC(gamma=2, C=1)
# SVM.fit(X_train, Y_train)
# SVM_pred = SVM.predict(X_test)
# write_predictions(SVM_pred, test_ids, 'predicted_SVM.csv')
# print 'SVC', categorization_accuracy('predicted_SVM.csv', 'actual_small.csv')
#
# # KNeighborsClassifier
# KN = KNeighborsClassifier()
# KN.fit(X_train, Y_train)
# KN_pred = KN.predict(X_test)
# write_predictions(KN_pred, test_ids, 'predicted_KN.csv')
# print 'KNeighborsClassifier', categorization_accuracy('predicted_KN.csv', 'actual_small.csv')
#
#
# # RandomForestClassifier 0.890524379025
# # warnings.warn("Variables are collinear")
# # QuadraticDiscriminantAnalysis 0.699172033119
# # AdaBoostClassifier 0.634774609016
# # GaussianNB 0.671573137075
# # DecisionTreeClassifier 0.866605335787
# # SVC 0.625574977001
# # KNeighborsClassifier 0.824287028519

# RandomForestClassifier 0.890524379025
# QuadraticDiscriminantAnalysis 0.699172033119
# AdaBoostClassifier 0.634774609016
# GaussianNB 0.671573137075
# DecisionTreeClassifier 0.866605335787
# SVC 0.625574977001
# KNeighborsClassifier 0.824287028519