from math import ceil
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale, StandardScaler
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# lab1
print("\nЛабораторная 1:")
df = pd.read_csv(filepath_or_buffer="titanic.csv")
passengersCount = df["PassengerId"].count()
SN = df[["Sex", "Name"]]
dfDescribe = df["Age"].describe()

print("1)", SN[SN["Sex"] == "female"].shape[0], df[df["Sex"] == "male"].shape[0])
print("2)", ceil(df[df["Survived"] == 0].shape[0] / passengersCount * 100))
print("3)", ceil(df[df["Pclass"] == 1].shape[0] / passengersCount * 100))
print("4)", dfDescribe[1], dfDescribe[5])
print("5)", df["SibSp"].corr(df["Parch"], method="pearson"))

dict = {}
maxVal = 0
for str in SN[SN["Sex"] == "female"].values:
    key = str[1].split(".")[1].strip()
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1

maxVal = 1
freqName = "A"
for key in dict.keys():
    if dict[key] > maxVal:
        maxVal = dict[key]
        freqName = key
print("6)", freqName)

# lab2
print("\nЛабораторная 2:")

data = pd.read_csv('titanic.csv', index_col='PassengerId',
                   usecols=['PassengerId', 'Pclass', 'Fare', 'Age', 'Sex', 'Survived'])

data['Sex'] = data['Sex'].factorize()[0]

data_no_na = data.dropna(axis=0)

feature_names = ['Pclass', 'Sex', 'Age', 'Fare']
features = np.array(data_no_na[feature_names].values)

response = np.array(data_no_na['Survived'].values)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(features, response)

importances = clf.feature_importances_
feature_importances_dict = {feature: importance for feature, importance in zip(feature_names, importances)}
feature_importances_dict = Counter(feature_importances_dict)
resulting_string = feature_importances_dict.most_common(1)[0][0] + ", " + feature_importances_dict.most_common(2)[1][0]
print("7)", resulting_string)

print("8)")
for predict in clf.predict([[1, 0, 10, 50], [2, 0, 50, 30], [3, 1, 10, 60], [1, 1, 50, 40]]):
    if (predict):
        print("Выжил")
    else:
        print("Умер")

# lab3.1
print("\nЛабораторная 3.1:")
wine = pd.read_csv("wine.data", usecols=['1'])

data = pd.read_csv("wine.data")
del data['1']

kf = KFold(n_splits=5, random_state=42, shuffle=True)

max = 0  # значение
kmax = 1  # индекс
for i in range(50):
    neigh = KNeighborsClassifier(n_neighbors=i + 1)
    val = cross_val_score(neigh, data, wine['1'], cv=kf, scoring='accuracy').mean()
    if val > max:
        kmax = i + 1
        max = val
print(max, kmax)

data = scale(data)

max = 0
kmax = 1
for i in range(50):
    neigh = KNeighborsClassifier(n_neighbors=i + 1)
    val = cross_val_score(neigh, data, wine['1'], cv=kf, scoring='accuracy').mean()
    if val > max:
        kmax = i + 1
        max = val
print(max, kmax)

# lab3.2
print("\nЛабораторная 3.2:")
california = scale(fetch_california_housing().data)

target = fetch_california_housing().target

kf = KFold(n_splits=5, random_state=42, shuffle=True)

min = 0
pmin = 0
for m in np.linspace(1, 10, 2):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=m)
    val = cross_val_score(neigh, california, target, cv=kf, scoring='neg_mean_squared_error').mean()
    if val < min:
        pmin = m
        min = val
print(min, pmin)

# lab4.1
print("\nЛабораторная 4.1:")
train_data = pd.read_csv('perceptron-train.csv')
test_data = pd.read_csv('perceptron-test.csv')

# Разделение данных на признаки и целевую переменную
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Обучение персептрона со стандартными параметрами и random_state=241
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# Подсчет качества классификатора на тестовой выборке
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Нормализация обучающей и тестовой выборки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение персептрона на нормализованных выборках
clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(X_train_scaled, y_train)

# Подсчет качества на тестовой выборке после нормализации
y_pred_scaled = clf_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# Разность между качеством на тестовой выборке после нормализации и до нее
difference = accuracy_scaled - accuracy

print("Разность качества:", difference)

# lab4.2
print("\nЛабораторная 4.2:")
data = pd.read_csv('apples_pears.csv')

#Построение изображения набора данных "Яблоки-груши" в виде точек на плоскости
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['target'], cmap='rainbow')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()

# Выделение матрицы признаков (Х) и вектора ответов (у)
X = data.iloc[:, :-1]
y = data['target']

# Обучение перцептрона на матрице признаков (Х) с учетом вектора ответов (у)
clf = Perceptron()
clf.fit(X, y)

# Построение изображения набора данных "Яблоки-груши" с учетом результатов классификации
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clf.predict(X), cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('Симметричность', fontsize=14)
plt.ylabel('Желтизна', fontsize=14)
plt.show()

# lab5.1
print("\nЛабораторная 5.1:")
from sklearn.svm import SVC

#Загрузка выборки из файла svm-data.csv
data = pd.read_csv('svm-data.csv', header=None)

#Построение изображения набора данных в виде точек на плоскости
plt.scatter(data.iloc[:, 1], data.iloc[:, 2], c=data.iloc[:, 0], cmap='viridis')
plt.show()

#Обучение классификатора с линейным ядром и параметром C=100000
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(data.iloc[:, 1:], data.iloc[:, 0])

#Нахождение номеров опорных объектов
support_vectors_idx = clf.support_

#Предсказание класса для новой точки
new_point = [[0.5, 0.5]]
predicted_class = clf.predict(new_point)

print(support_vectors_idx, predicted_class)

# lab5.2
print("\nЛабораторная 5.2:")
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

#Загрузка объектов из новостного датасета 20newsgroups
categories = ['comp.sys.ibm.pc.hardware', 'talk.religion.misc']
data = fetch_20newsgroups(subset='all', categories=categories)

#Вычисление TF-IDF-признаков для всех текстов
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data.data)
y = data.target

#Подбор минимального лучшего параметра C с помощью кросс-валидации
param_grid = {'C': np.logspace(-5, 5, 11)}
clf = SVC(kernel='linear', random_state=241)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
best_C = grid_search.best_params_['C']

#Обучение SVM по всей выборке с лучшим параметром C
clf_best = SVC(kernel='linear', C=best_C, random_state=241)
clf_best.fit(X, y)

#Нахождение 10 слов с наибольшим по модулю весом
feature_names = np.array(tfidf.get_feature_names())
weights = np.abs(clf_best.coef_.toarray()[0])
top_10_words_idx = np.argsort(-weights)[:10]
top_10_words = feature_names[top_10_words_idx]

print(top_10_words)
