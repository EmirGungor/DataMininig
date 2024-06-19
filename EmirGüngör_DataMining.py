#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('/mnt/data/wine.data', header=None, names=columns)
wine_data.head()

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = '/mnt/data/wine.data'  # Yüklenen dosyanın yolu
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(15, 10))
wine_data.hist(bins=20, figsize=(15, 10), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(15, 10))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(knn, X_test, y_test, cmap='Blues')
plt.title('k-NN Karışıklık Matrisi')
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(nb, X_test, y_test, cmap='Blues')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(dt, X_test, y_test, cmap='Blues')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = 'wine.data'  # Yüklenen dosyanın yolu
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())


# In[4]:


# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(15, 10))
wine_data.hist(bins=20, figsize=(15, 10), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(15, 10))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# In[5]:


# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(knn, X_test, y_test, cmap='Blues')
plt.title('k-NN Karışıklık Matrisi')
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(nb, X_test, y_test, cmap='Blues')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(6, 6))
plot_confusion_matrix(dt, X_test, y_test, cmap='Blues')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[7]:


# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(10, 8))
plot_confusion_matrix(knn, X_test, y_test, cmap='Blues')
plt.title('k-NN Karışıklık Matrisi')
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(10, 8))
plot_confusion_matrix(nb, X_test, y_test, cmap='Blues')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(10, 8))
plot_confusion_matrix(dt, X_test, y_test, cmap='Blues')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[8]:


# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(20, 8))
plot_confusion_matrix(knn, X_test, y_test, cmap='Blues')
plt.title('k-NN Karışıklık Matrisi')
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(20, 8))
plot_confusion_matrix(nb, X_test, y_test, cmap='Blues')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(20, 8))
plot_confusion_matrix(dt, X_test, y_test, cmap='Blues')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = '/mnt/data/wine.data'  # Yüklenen dosyanın yolu
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = '/mnt/data/wine.data'  # Yüklenen dosyanın yolu
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = '/mnt/data/wine.data'  # Yüklenen dosyanın doğru yolu
columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = '/mnt/data/wine.data'  # Yüklenen dosyanın doğru yolu
columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = 'Desktop/wine.data'  # Yüklenen dosyanın doğru yolu
columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = 'C:/Users/E/Desktop/wine.data'  # Yüklenen dosyanın doğru yolu
columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(12, 12))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[15]:


plt.figure(figsize=(20, 15))
wine_data.hist(bins=20, figsize=(20, 15), grid=False)
plt.show()


# In[16]:


plt.figure(figsize=(20, 15))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# In[17]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[18]:


plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# In[19]:


plt.figure(figsize=(100, 100))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# In[20]:


plt.figure(figsize=(2, 2))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# In[21]:


plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=1000)
plt.yticks(rotation=1000)
plt.show()


# In[22]:


plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90, fontsize=10)  # Yazı tipi boyutunu küçültme
plt.yticks(rotation=0, fontsize=10)   # Yazı tipi boyutunu küçültme
plt.gca().set_aspect('auto')  # Eksenler arası boşlukları arttırma
plt.show()


# In[23]:


# k-NN için karışıklık matrisi
plt.figure(figsize=(15, 15))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=90, fontsize=8)  # Yazı tipi boyutunu daha da küçültme
plt.yticks(rotation=0, fontsize=8)   # Yazı tipi boyutunu daha da küçültme
plt.gca().set_aspect('auto')  # Eksenler arası boşlukları arttırma
plt.tight_layout()  # Yerleşimi sıkıştırarak düzenleme
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(15, 15))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(15, 15))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yollarını kontrol edelim ve yükleyelim
wine_data_path = 'C:/Users/YourUsername/Desktop/wine.data'  # Yüklenen dosyanın yolu
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(15, 10))
wine_data.hist(bins=20, figsize=(15, 10), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(15, 10))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yolunu doğru bir şekilde belirleyin
wine_data_path = 'C:\Users\EmirG\OneDrive\Desktop\wine.data'  # Burada 'YourActualUsername' kısmını kendi kullanıcı adınızla değiştirin
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setinin ilk 5 satırını gösterme
print(wine_data.head())

# Veri setinin istatistiksel özeti
print(wine_data.describe())

# Eksik değer kontrolü
print(wine_data.isnull().sum())

# Özelliklerin dağılımını görselleştirme
plt.figure(figsize=(15, 10))
wine_data.hist(bins=20, figsize=(15, 10), grid=False)
plt.show()

# Özellikler arasındaki ilişkileri ısı haritası ile gösterme
plt.figure(figsize=(15, 10))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes modeli oluşturma ve değerlendirme
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Karar Ağacı modeli oluşturma ve değerlendirme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Naive Bayes için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Karar Ağacı için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap='Blues', values_format='.2f')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# Modellerin performans raporları
print("k-NN Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))
print("Naive Bayes Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))
print("Karar Ağacı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_dt))


# In[26]:


plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dosya yolunu belirleyin
wine_data_path = 'C:\\Users\\EmirG\\OneDrive\\Desktop\\wine.data'

# Veri setini yükleme
columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
]

# Veri setini yükleme
wine_data = pd.read_csv(wine_data_path, header=None, names=columns)

# Veri setini eğitim ve test olarak ayırma
X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modeli oluşturma ve değerlendirme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# k-NN için karışıklık matrisi
plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues', values_format='.2f')
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[28]:


X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[29]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[31]:


plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(12, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('k-NN Karışıklık Matrisi')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[33]:


columns = [
    'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 
    'Hue', 'diluted_wines', 'Proline'
]


# In[34]:


X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[35]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[36]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[37]:


plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('k-NN Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[38]:


plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('k-NN Karışıklık Matrisi', fontsize=65)
plt.xticks(rotation=65, ha='right', fontsize=35)
plt.yticks(fontsize=35)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[39]:


X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[41]:


plt.figure(figsize=(20, 20))
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap='Blues', values_format='d', ax=plt.gca())


# In[42]:


for i in range(cm_knn.shape[0]):
    for j in range(cm_knn.shape[1]):
        plt.text(j, i, format(cm_knn[i, j], 'd'), ha="center", va="center", color="black", fontsize=20, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.title('k-NN Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[43]:


from IPython.core.display import display, HTML

# Çıktı hücresinin yüksekliğini ve genişliğini ayarlama
display(HTML("<style>.output {height: 1000px; width: 100% !important;}</style>"))


# In[44]:


from IPython.core.display import display, HTML

# Çıktı hücresinin yüksekliğini ve genişliğini ayarlama
display(HTML("<style>.output {height: 2000px; width: 100% !important;}</style>"))


# In[45]:


plt.title('k-NN Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=15)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# In[46]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)


# In[47]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# In[48]:


plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[49]:


plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[50]:


plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[51]:


plt.figure(figsize=(20, 20))
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap='Blues', values_format='.2f')
plt.title('Naive Bayes Karışıklık Matrisi', fontsize=20)
plt.xticks(rotation=45, fontsize=5)
plt.yticks(rotation=0, fontsize=5)
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()


# In[52]:


def plot_confusion_matrix(cm, title, fontsize=20):
    plt.figure(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='.2f')
    plt.title(title, fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.gca().set_aspect('auto')
    plt.tight_layout()
    plt.show()


# In[53]:


X = wine_data.drop('Proline', axis=1)
y = wine_data['Proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[54]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[55]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)


# In[56]:


cm_nb = confusion_matrix(y_test, y_pred_nb)
plot_confusion_matrix(cm_nb, 'Naive Bayes Karışıklık Matrisi')


# In[57]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(cm, title, fontsize=20):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax)
    plt.title(title, fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.gca().set_aspect('auto')
    plt.tight_layout()

    # Her bir hücredeki değeri ayarlamak
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

# Naive Bayes için karışıklık matrisi
cm_nb = confusion_matrix(y_test, y_pred_nb)
plot_confusion_matrix(cm_nb, 'Naive Bayes Karışıklık Matrisi')


# In[58]:


# Karar Ağacı için karışıklık matrisi
cm_dt = confusion_matrix(y_test, y_pred_dt)
plot_confusion_matrix(cm_dt, 'Karar Ağacı Karışıklık Matrisi')


# In[59]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(cm, title, fontsize=20):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax)
    plt.title(title, fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.gca().set_aspect('auto')
    plt.tight_layout()

    # Her bir hücredeki değeri ayarlamak
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

# Karar Ağacı için karışıklık matrisi
cm_dt = confusion_matrix(y_test, y_pred_dt)
plot_confusion_matrix(cm_dt, 'Karar Ağacı Karışıklık Matrisi')


# In[60]:


# k-NN için karışıklık matrisi
cm_knn = confusion_matrix(y_test, y_pred_knn)
plot_confusion_matrix(cm_knn, 'k-NN Karışıklık Matrisi')


# In[61]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(cm, title, fontsize=20):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax)
    plt.title(title, fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.gca().set_aspect('auto')
    plt.tight_layout()

    # Her bir hücredeki değeri ayarlamak
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

# k-NN için karışıklık matrisi
cm_knn = confusion_matrix(y_test, y_pred_knn)
plot_confusion_matrix(cm_knn, 'k-NN Karışıklık Matrisi')


# In[62]:


from sklearn.metrics import classification_report, accuracy_score

# k-NN doğruluk ve sınıflandırma raporu
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)
print(f"k-NN Doğruluk: {accuracy_knn}")
print(f"k-NN Sınıflandırma Raporu:\n{report_knn}")

# Naive Bayes doğruluk ve sınıflandırma raporu
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)
print(f"Naive Bayes Doğruluk: {accuracy_nb}")
print(f"Naive Bayes Sınıflandırma Raporu:\n{report_nb}")

# Karar Ağacı doğruluk ve sınıflandırma raporu
accuracy_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)
print(f"Karar Ağacı Doğruluk: {accuracy_dt}")
print(f"Karar Ağacı Sınıflandırma Raporu:\n{report_dt}")


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Data Information
wine_data.info()

# Statistical Summary
wine_data_description = wine_data.describe()
print(wine_data_description)

# Creating a table figure
fig, ax = plt.subplots(figsize=(12, 6))  # set size frame
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=wine_data_description.values,
                 colLabels=wine_data_description.columns,
                 rowLabels=wine_data_description.index,
                 cellLoc='center', loc='center')

# Displaying the table
plt.title("Wine Dataset Statistical Summary")
plt.show()

# Splitting the data
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Making predictions
y_pred = knn.predict(X_test_scaled)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[2]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modeli oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# kNN modelini yeniden eğitme (PCA sonrası)
knn.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Görselleştirme
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('kNN Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()


# In[3]:


from matplotlib.colors import ListedColormap

# Renklendirme haritalarını tekrar tanımlayalım
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()


# In[4]:


# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)

accuracy_nb


# In[5]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[6]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[7]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[8]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[9]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[10]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Decision Tree modelini yeniden eğitme (PCA sonrası)
dt.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Decision Tree Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[11]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Naive Bayes modelini yeniden eğitme (PCA sonrası)
nb.fit(X_train_pca, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary with PCA Components')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_pca)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[12]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma
nb = GaussianNB()

# Öğrenme eğrisini hesaplama
train_sizes, train_scores, valid_scores = learning_curve(nb, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Öğrenme eğrisi için ortalama ve standart sapma hesaplama
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Öğrenme eğrisi grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Eğitim skoru")
plt.plot(train_sizes, valid_mean, 'o-', color="g", label="Doğrulama skoru")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, color="g", alpha=0.1)

plt.title('Naive Bayes Öğrenme Eğrisi')
plt.xlabel('Eğitim Seti Boyutu')
plt.ylabel('Skor')
plt.legend(loc="best")
plt.grid()
plt.show()


# In[13]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma
dt = DecisionTreeClassifier(random_state=42)

# Öğrenme eğrisini hesaplama
train_sizes, train_scores, valid_scores = learning_curve(dt, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Öğrenme eğrisi için ortalama ve standart sapma hesaplama
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Öğrenme eğrisi grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Eğitim skoru")
plt.plot(train_sizes, valid_mean, 'o-', color="g", label="Doğrulama skoru")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, color="g", alpha=0.1)

plt.title('Karar Ağacı Öğrenme Eğrisi')
plt.xlabel('Eğitim Seti Boyutu')
plt.ylabel('Skor')
plt.legend(loc="best")
plt.grid()
plt.show()


# In[14]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Özellikleri ve hedef değişkeni ayırma
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma (farklı max_depth değerleriyle)
max_depth_values = [1, 3, 5, 7, 9]
plt.figure(figsize=(14, 8))

for max_depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    train_sizes, train_scores, valid_scores = learning_curve(dt, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Öğrenme eğrisi için ortalama ve standart sapma hesaplama
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    # Öğrenme eğrisi grafiği
    plt.plot(train_sizes, train_mean, 'o-', label=f'Eğitim skoru (max_depth={max_depth})')
    plt.plot(train_sizes, valid_mean, 'o-', label=f'Doğrulama skoru (max_depth={max_depth})')

plt.title('Karar Ağacı Öğrenme Eğrisi (max_depth ile)')
plt.xlabel('Eğitim Seti Boyutu')
plt.ylabel('Skor')
plt.legend(loc="best")
plt.grid()
plt.show()


# In[15]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Magnesium', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('Naive Bayes Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[16]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Magnesium', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('kNN Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[17]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Magnesium', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('kNN Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[18]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Magnesium', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFFFFF', '#FFFFFF', '#FFFFFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('Decision Tree Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[19]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Alcohol', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFFFFF', '#FFFFFF', '#FFFFFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('Decision Tree Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[20]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Alcohol', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFFFFF', '#FFFFFF', '#FFFFFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('Decision Tree Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[21]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece magnezyum ve malic asit özelliklerini seçme
X = wine_data[['Alcohol', 'Malic_acid']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Magnezyum (Standardize)')
plt.ylabel('Malic Acid (Standardize)')
plt.title('kNN Decision Boundary with Magnesium and Malic Acid')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[22]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[23]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturma ve eğitme
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Naive Bayes Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Naive Bayes modeli ile test seti üzerinde tahmin yapma
y_pred_nb = nb.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Doğruluk Oranı: {accuracy_nb * 100:.2f}%')


# In[24]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(


# In[25]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(


# In[26]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Decision Tree Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[27]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[28]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[29]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[30]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma ve eğitme
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Decision Tree Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# Decision Tree modeli ile test seti üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Doğruluk Oranı: {accuracy_dt * 100:.2f}%')


# In[31]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluşturma
dt = DecisionTreeClassifier(random_state=42)

# Hiperparametre arama için parametre ızgarası
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Grid Search kullanarak en iyi modeli bulma
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# En iyi parametreleri gösterme
print(f"En İyi Parametreler: {grid_search.best_params_}")

# En iyi model ile tahmin yapma
best_dt = grid_search.best_estimator_
y_pred_dt = best_dt.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'En İyi Karar Ağacı Doğruluk Oranı: {accuracy_dt * 100:.2f}%')

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = best_dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Decision Tree Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()


# In[32]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[33]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[34]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[35]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[36]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[37]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[38]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[39]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[40]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .03  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[41]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .05  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[42]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[43]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('kNN Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train))
plt.show()

# kNN modeli ile test seti üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Doğruluk Oranı: {accuracy_knn * 100:.2f}%')


# In[1]:


# Gerekli kütüphaneleri yükleme
from sklearn.ensemble import RandomForestClassifier

# Rastgele Orman modelini oluşturma ve eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rf = Z_rf.reshape(xx.shape)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_rf, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter_rf = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Random Forest Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter_rf.legend_elements()[0], labels=set(y_train))
plt.show()

# Rastgele Orman modeli ile test seti üzerinde tahmin yapma
y_pred_rf = rf.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Rastgele Orman Doğruluk Oranı: {accuracy_rf * 100:.2f}%')


# In[2]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('/mnt/data/wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rastgele Orman modelini oluşturma ve eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rf = Z_rf.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_rf, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter_rf = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Random Forest Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter_rf.legend_elements()[0], labels=set(y_train))
plt.show()

# Rastgele Orman modeli ile test seti üzerinde tahmin yapma
y_pred_rf = rf.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Rastgele Orman Doğruluk Oranı: {accuracy_rf * 100:.2f}%')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rastgele Orman modelini oluşturma ve eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rf = Z_rf.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_rf, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter_rf = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Random Forest Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter_rf.legend_elements()[0], labels=set(y_train))
plt.show()

# Rastgele Orman modeli ile test seti üzerinde tahmin yapma
y_pred_rf = rf.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Rastgele Orman Doğruluk Oranı: {accuracy_rf * 100:.2f}%')


# In[4]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('/mnt/data/wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lojistik Regresyon modelini oluşturma ve eğitme
lr = LogisticRegression(random_state=42, multi_class='ovr')
lr.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_lr, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter_lr = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Logistic Regression Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter_lr.legend_elements()[0], labels=set(y_train))
plt.show()

# Lojistik Regresyon modeli ile test seti üzerinde tahmin yapma
y_pred_lr = lr.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Lojistik Regresyon Doğruluk Oranı: {accuracy_lr * 100:.2f}%')


# In[5]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)

# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]
y = wine_data['Class']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lojistik Regresyon modelini oluşturma ve eğitme
lr = LogisticRegression(random_state=42, multi_class='ovr')
lr.fit(X_train_scaled, y_train)

# Karar sınırlarını görselleştirme
h = .02  # meshgrid adımı
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)

# Renklendirme haritalarını tanımlama
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_lr, cmap=cmap_light)

# Eğitim verilerini scatter plot ile görselleştirme
scatter_lr = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Alcohol (Standardize)')
plt.ylabel('Magnesium (Standardize)')
plt.title('Logistic Regression Decision Boundary with Alcohol and Magnesium')
plt.legend(handles=scatter_lr.legend_elements()[0], labels=set(y_train))
plt.show()

# Lojistik Regresyon modeli ile test seti üzerinde tahmin yapma
y_pred_lr = lr.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Lojistik Regresyon Doğruluk Oranı: {accuracy_lr * 100:.2f}%')


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükleme
data_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
    'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv('wine.data', header=None, names=data_columns)


# Sadece alkol ve magnezyum özelliklerini seçme
X = wine_data[['Alcohol', 'Magnesium']]

# Normalizasyon öncesi veri görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(X['Alcohol'], bins=20, color='blue', alpha=0.7)
plt.title('Alkol Dağılımı (Normalizasyon Öncesi)')
plt.xlabel('Alkol')
plt.ylabel('Frekans')

plt.subplot(1, 2, 2)
plt.hist(X['Magnesium'], bins=20, color='green', alpha=0.7)
plt.title('Magnezyum Dağılımı (Normalizasyon Öncesi)')
plt.xlabel('Magnezyum')
plt.ylabel('Frekans')

plt.tight_layout()
plt.show()


# In[7]:


from sklearn.preprocessing import MinMaxScaler

# Min-Max normalizasyonu
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Normalizasyon sonrası veri görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(X_normalized[:, 0], bins=20, color='blue', alpha=0.7)
plt.title('Alkol Dağılımı (Normalizasyon Sonrası)')
plt.xlabel('Alkol')
plt.ylabel('Frekans')

plt.subplot(1, 2, 2)
plt.hist(X_normalized[:, 1], bins=20, color='green', alpha=0.7)
plt.title('Magnezyum Dağılımı (Normalizasyon Sonrası)')
plt.xlabel('Magnezyum')
plt.ylabel('Frekans')

plt.tight_layout()
plt.show()


# In[ ]:




