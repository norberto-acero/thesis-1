#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore'); warnings.simplefilter('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train, test = train_test_split(pd.read_excel('Data.xlsx')
                                , stratify=pd.read_excel('Data.xlsx')['CLASE']
                                , random_state=2023
                                , train_size=0.7
                                )
sabana_paleta = ['#00205B', '#DCA46C','#FF9900','#D0D2D2','#D0D3D4','#68B5E2','#8E2C48','#1374AA','#D29F13','#CB2929','#96D600']
paleta = {'CORRIENTE':'#00205B', 'VENCIDO':'#DCA46C'}
#%% Construcción de la tabla dinamica para gráfico:
t = train.groupby(['CLASE','ESTRATO']).size().reset_index(name='CANTIDAD')
t['TCLASE'] = t.groupby('CLASE')['CANTIDAD'].transform('sum')
t['PORCENTAJE'] = (t['CANTIDAD']/t['TCLASE'])
t

#%% - Gráficos barras para categóricas:
for i in train.select_dtypes(include='object').columns.tolist():
    if i != 'CLASE':
        sbn.set_theme(style="darkgrid")
        t = train.groupby(['CLASE',i]).size().reset_index(name='CANTIDAD')
        t['TCLASE'] = t.groupby('CLASE')['CANTIDAD'].transform('sum')
        t['PORCENTAJE'] = (t['CANTIDAD']/t['TCLASE'])
        fig, ax = plt.subplots(figsize=(12,7), sharey=False)
        for c in t['CLASE'].unique():
            data = t[t['CLASE']==c]
            sbn.barplot(data=t, hue=t[i], y=t['PORCENTAJE'], x=t['CLASE'], palette=sabana_paleta, edgecolor='white')
        ax.set_xlabel(i)
        ax.set_ylabel(f'Porcentaje de {i} por CLASE')
#!        ax.set_title('TITULO')
#!        ax.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'center', 
            xytext = (0, 10), 
            textcoords = 'offset points')
#!        plt.xticks(rotation=90)
        plt.show()

#%% Gráfico Pie para la clase:
sbn.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(15, 5), subplot_kw=dict(aspect="equal"))
data = train['CLASE'].value_counts().values
Etiquetas = train['CLASE'].value_counts().index
def func(pct, allvals):
        absoluto = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absoluto:d} g)"
wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data), textprops=dict(color="w"), colors=sabana_paleta)
ax.legend(wedges, Etiquetas, title="Etiquetas", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=8, weight="bold")
plt.show()

#%% Gráficos estadisticos para las numéricas:
for i in train.select_dtypes(include=np.number).columns.tolist():
    fig, axs = plt.subplots(1,3, figsize=(15,5), sharey=False)
    ax=sbn.histplot(train, x=i, kde=True, hue='CLASE', ax=axs[0], edgecolor='black', bins=25, palette=paleta)
    ax.lines[0].set_color('red')
    sbn.boxplot(train, x=i, hue='CLASE', ax=axs[1], medianprops={"color":"red"}, palette=paleta)
    sbn.violinplot(train, x=i, hue='CLASE', ax=axs[2], inner=None, palette=paleta)
    sbn.boxenplot(train, x=i, color='red', width=0.03, ax=axs[2])
    fig.subplots_adjust(wspace=0.1)
    plt.show()

#%% Correlación entre variables numéricas
sbn.set_theme(style="darkgrid")
pd.set_option('display.float_format', lambda x: '%.0f' % x)
sbn.heatmap(train.select_dtypes(include=(np.number)).corr(), annot=True, cmap='rocket_r')

#%% emparejamiento de variables numericas
sbn.pairplot(train, hue='CLASE', palette=paleta)

#%% Gráfica MCA de prince:
sbn.set_theme(style="darkgrid")
mca = pd.read_excel('MCA_.xlsx')
g1=sbn.barplot(mca['% of var.'], color="#00205B")
g1.set(xticklabels=[])
g1.set(xlabel=None)
plt.show()

#%% PCA:
train = train.dropna() #TODO se eliminan las filas con valores faltantes
train2 = train.copy() #TODO se copia el DF para que no se pierda.
cat = train2.select_dtypes(include=['object']).columns #TODO se separan las variables categóricas.
num = train2.select_dtypes(np.number).columns #TODO ...de las numéricas.
cod = pd.get_dummies(train2, columns=cat) #TODO se binarizan las variables categoricas.
sc = StandardScaler() #TODO se llama la función del estandarizador
sco = pd.DataFrame(sc.fit_transform(cod), columns=cod.columns) #TODO se arma el nuevo dataframe con la estandarización
pca = PCA(n_components=2) #TODO se llama la función PCA de scikilearn.
pca.fit_transform(sco) #TODO se adaptan los datos al PCA.
pcdata = pca.transform(sco)
C= train2.dropna().filter(items=['CLASE'])
P= pd.DataFrame(pcdata)
C.reset_index(drop=True, inplace=True)
P.reset_index(drop=True, inplace=True)
A= pd.concat([C,P], axis=1)
ax=sbn.scatterplot(data=A, x=0, y=1,  hue='CLASE', palette=sabana_paleta)
ax.set_ylabel('')
ax.set_xlabel('')

#%% Modelo cerrado:
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, make_scorer
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Data.xlsx')

X = df.drop('CLASE', axis=1)
X = pd.get_dummies(X)
y = df['CLASE']

numeric_features = X.select_dtypes(include='float').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

#TODO Crear un imputador para las variables numéricas
numeric_imputer = SimpleImputer(strategy='median')

#TODO Crear un imputador para las variables categóricas
categorical_imputer = SimpleImputer(strategy='most_frequent')

#TODO Crear un escalador para la normalización de las variables numéricas
scaler = MinMaxScaler()

#TODO Crear el modelo HistGradientBoostingClassifier con los hiperparámetros especificados
classifier = HistGradientBoostingClassifier(
     early_stopping=True
    ,l2_regularization=0.00204376553180705
    ,learning_rate=0.022346785733375032
    ,loss='log_loss'
    ,max_iter=512
    ,max_leaf_nodes=5
    ,min_samples_leaf=1
    ,n_iter_no_change=16
    ,validation_fraction=None
    ,warm_start=True
)

#TODO Crear un ColumnTransformer para aplicar las transformaciones adecuadas a las columnas numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_imputer, numeric_features)
        ,('cat', categorical_imputer, categorical_features)
        ,('scaler_C', scaler, categorical_features)
        ,('scaler_N', scaler, numeric_features)
        ])

#TODO Crear el pipeline completo con preprocesamiento y clasificador
pipeline = Pipeline(steps=[('preprocessor', preprocessor)
                           ,('classifier', classifier)
                           ])

#TODO Ajustar el modelo al conjunto de entrenamiento
pipeline.fit(X_train, y_train)

#TODO Realizar predicciones en el conjunto de prueba
y_pred = pipeline.predict(X_test)

#TODO Calcular la precisión del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#TODO Realizar validación cruzada para evaluar el rendimiento del modelo
cv_accuracy = cross_val_score(pipeline, X, y, cv=5, scoring=make_scorer(accuracy_score))
print("Cross-Validation Accuracy:", cv_accuracy.mean())

#%% Curva ROC   
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
X = A.drop(['CLASE'], axis=1).values
y = A['CLASE'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2023)
CLS = HistGradientBoostingClassifier()
CLS.fit(X_train, y_train)
