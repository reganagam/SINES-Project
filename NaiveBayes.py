# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor datasetnya
dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Mengubah data string menjadi int
from sklearn.preprocessing import LabelEncoder as LE
le = LE()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['family_history_with_overweight'] = le.fit_transform(dataset['family_history_with_overweight'])
dataset['FAVC'] = le.fit_transform(dataset['FAVC'])
dataset['CAEC'] = le.fit_transform(dataset['CAEC'])
dataset['SMOKE'] = le.fit_transform(dataset['SMOKE'])
dataset['SCC'] = le.fit_transform(dataset['SCC'])
dataset['CALC'] = le.fit_transform(dataset['CALC'])
dataset['MTRANS'] = le.fit_transform(dataset['MTRANS'])
dataset['NObeyesdad'] = le.fit_transform(dataset['NObeyesdad'])

# Menentukan variabel x dan y
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values
print(dataset)

# Menjadi dataset ke dalam Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
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