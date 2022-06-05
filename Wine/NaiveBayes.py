# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('Dataset/wine.data')
X = dataset.iloc[:, 1:14].values
y = dataset.iloc[:, 0].values
 
# Menjadi dataset ke dalam Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n\n"+"="*50)
print("Dataset")
print(dataset)
 
# Membuat model Naive Bayes terhadap Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred = classifier.predict(X_test)
 
# Membuat confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\n\n"+"="*50)
print("Matriks Kekeliruan")
print(cm)

# Mengecek tingkat keakurasian
from sklearn.metrics import accuracy_score
print("\n\n"+"="*50)
print("Tingkat Keakurasian : ", accuracy_score(y_test,y_pred))

# Membandingkan Nilai Asli dan Nilai Prediksi
print("\n\n"+"="*50)
print(pd.DataFrame({"Nilai Asli : " : y_test, "Nilai Prediksi : " : y_pred}))