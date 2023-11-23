#%%
import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sbn
from sklearn.model_selection import train_test_split, LearningCurveDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

df = pd.read_excel('/Users/norbertoacero/Proyects/thesis-1/Data.xlsx')
X = pd.get_dummies(df.drop('CLASE', axis=1).dropna(), dtype='float')
y = df.dropna()['CLASE']
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2023, test_size=0.3, stratify=df.dropna()['CLASE'])

clf = GradientBoostingClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_pred, y_test)
rocauc = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])


# fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:,1], pos_label='CORRIENTE')
# plt.plot(fpr, tpr)
# sbn.heatmap(cm, annot=True, fmt='.0f', cmap='rocket_r')

RocCurveDisplay.from_estimator(clf, x_test, y_test, pos_label='CORRIENTE')
LearningCurveDisplay.from_estimator(clf, x_test, y_test)
ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)

#%%
plt.plot(clf.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')

#%%
pca = PCA(n_components=2).fit_transform(x_train)
C= y_train
P= pd.DataFrame(pca)
C.reset_index(drop=True, inplace=True)
P.reset_index(drop=True, inplace=True)
A= pd.concat([C,P], axis=1)
# ax=sbn.scatterplot(data=A, x=0, y=1,  hue='CLASE')
# ax.set_ylabel('')
# ax.set_xlabel('')

sbn.pairplot(data=A, hue='CLASE')