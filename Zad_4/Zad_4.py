import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Wczytywanie danych z pliku CSV
df = pd.read_csv(r'C:\Users\USER\PycharmProjects\POI-repozytorium\Zad_3\zbiór_wektorów_danych.csv', sep=',')

# Wybór tylko kolumn zawierających cechy
X = df.drop(['plik', 'kategoria'], axis=1).astype('float')

# Wybór kolumny kategorii jako etykiety
y = df['kategoria']

# Kodowanie etykiet
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

# Kodowanie 1-z-n dla etykiet
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Podział danych na część treningową i testową
X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True, verbose=0)

# Testowanie modelu
y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Wyliczenie macierzy pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
