import os, cv2
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def eks_cechy_tekstury(ścieżka_katalogu, odległości=[1, 3, 5], kąty=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    lista_cech = []

    pliki = os.listdir(ścieżka_katalogu)
    if not pliki:
        print("Brak obrazów do przetworzenia")
        return pd.DataFrame()

    for indeks, nazwa_pliku in enumerate(pliki):
        if not nazwa_pliku.endswith('.jpg'):
            continue


        ścieżka_obrazu = os.path.join(ścieżka_katalogu, nazwa_pliku)
        foto = io.imread(ścieżka_obrazu)
        if foto is None or foto.size == 0:
            print(f"Nie załadowano obrazu: {nazwa_pliku}")
            continue


        if len(foto.shape) == 3:
            foto_gray = color.rgb2gray(foto)
        else:
            foto_gray = foto

        foto_gray = img_as_ubyte(foto_gray)
        foto_gray //= 4


        glcm = graycomatrix(foto_gray, odległości, kąty, 64, symmetric=True, normed=True)
        cechy = {'plik': nazwa_pliku, 'kategoria': nazwa_pliku.split('_')[0]}

        for cecha in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
            for dist in odległości:
                for ang in kąty:
                    cechy[f'{cecha}_d{dist}_a{int(np.degrees(ang))}'] = graycoprops(glcm, cecha)[
                        odległości.index(dist), kąty.index(ang)]

        lista_cech.append(cechy)

    df_cech = pd.DataFrame(lista_cech)
    print(f"Liczba próbek dla których wyekstrahowano cechy: {len(df_cech)} ")
    return df_cech

def klasyfikacja_cech(df_cech):
    if df_cech.empty or 'kategoria' not in df_cech:
        print("Brak danych do klasyfikacji lub brakująca kategoria 'kategoria'.")
        return

    X = df_cech.drop(['plik', 'kategoria'], axis=1)
    y = df_cech['kategoria']

    if len(set(y)) < 2:
        print("Niewystarczająca liczba klas do klasyfikacji.")
        return

    X_treningowe, X_testowe, y_treningowe, y_testowe = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_treningowe, y_treningowe)
    y_pred = knn.predict(X_testowe)

    dokładność = accuracy_score(y_testowe, y_pred)
    print(f"Dokładność algorytmu KNN: {dokładność:.4f}")


def przytnij_i_zapisz_obrazy(wejście, wyjście, rozmiar_przycięcia=(128, 128)):
    if not os.path.exists(wyjście):
        os.makedirs(wyjście)

    print("Rozpoczynanie przycinania obrazów...")
    całkowita_liczba_przycięć = 0


    for ścieżka_katalogu in wejście:
        pliki = [f for f in os.listdir(ścieżka_katalogu) if f.endswith('.jpg')]


        for nazwa_pliku in pliki:
            ścieżka_obrazu = os.path.join(ścieżka_katalogu, nazwa_pliku)
            obraz = cv2.imread(ścieżka_obrazu)
            if obraz is None:
                print(f"Nie udało się załadować obrazu {nazwa_pliku} z {ścieżka_katalogu}")
                continue
            h, w, _ = obraz.shape


            id_przycięcia = 0
            for y in range(0, h - rozmiar_przycięcia[1] + 1, rozmiar_przycięcia[1]):
                for x in range(0, w - rozmiar_przycięcia[0] + 1, rozmiar_przycięcia[0]):
                    przycięty_obraz = obraz[y:y + rozmiar_przycięcia[1], x:x + rozmiar_przycięcia[0]]
                    ścieżka_przyciętego_obrazu = os.path.join(wyjście,
                                                              f"{os.path.splitext(nazwa_pliku)[0]}_{id_przycięcia}.jpg")
                    cv2.imwrite(ścieżka_przyciętego_obrazu, przycięty_obraz)
                    id_przycięcia += 1
                    całkowita_liczba_przycięć += 1
            print(f"Przycięto {id_przycięcia} obrazów z {ścieżka_obrazu}")

    print(f"Całkowita liczba przyciętych obrazów: {całkowita_liczba_przycięć}")

wejście= [
    r"C:\Users\USER\PycharmProjects\pythonProject1\wej\Gres",
    r"C:\Users\USER\PycharmProjects\pythonProject1\wej\Tynk",
    r"C:\Users\USER\PycharmProjects\pythonProject1\wej\Laminat"
]
wyjście = r"C:\Users\USER\PycharmProjects\POI-repozytorium\Zad_3\wyjście"

przytnij_i_zapisz_obrazy(wejście, wyjście)

df_cech = eks_cechy_tekstury(wyjście)

df_cech.to_csv('zbiór_wektorów_danych.csv', index=False)

klasyfikacja_cech(df_cech)