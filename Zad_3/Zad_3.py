import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def crop_and_save_images(source_folders, dest_folder, crop_size=(128, 128)):
    # Sprawdź, czy folder docelowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    print("Rozpoczynanie przycinania obrazów...")
    total_crops = 0  # Licznik przyciętych obrazów

    # Przetwarzaj każdy katalog źródłowy
    for source_folder in source_folders:
        # Znajdź wszystkie pliki .jpg w katalogu źródłowym
        files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

        # Przetwarzaj każdy plik obrazu
        for file_name in files:
            img_path = os.path.join(source_folder, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Nie udało się załadować obrazu {file_name} z {source_folder}")
                continue
            h, w, _ = img.shape  # Pobierz wymiary obrazu

            # Przycinanie obrazu
            crop_id = 0
            for y in range(0, h - crop_size[1] + 1, crop_size[1]):
                for x in range(0, w - crop_size[0] + 1, crop_size[0]):
                    crop_img = img[y:y + crop_size[1], x:x + crop_size[0]]
                    crop_img_path = os.path.join(dest_folder, f"{os.path.splitext(file_name)[0]}_{crop_id}.jpg")
                    cv2.imwrite(crop_img_path, crop_img)
                    crop_id += 1
                    total_crops += 1
            print(f"Przycięto {crop_id} obrazów z {img_path}")
    print(f"Całkowita liczba przyciętych obrazów: {total_crops}")


def extract_texture_features(folder_path, distances=[1, 3, 5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    features_list = []
    print("Ekstrakcja cech tekstur...")

    files = os.listdir(folder_path)
    if not files:
        print("Brak obrazów do przetworzenia w katalogu.")
        return pd.DataFrame()

    # Przetwarzaj każdy obraz w katalogu
    for index, file_name in enumerate(files):
        if file_name.endswith('.jpg'):
            print(f"Ładowanie {file_name} ({index + 1}/{len(files)})...")
            img_path = os.path.join(folder_path, file_name)
            img = io.imread(img_path)
            if img is None or img.size == 0:
                print(f"Nie udało się załadować obrazu {file_name}")
                continue

            # Konwersja obrazu do skali szarości
            if len(img.shape) == 3:
                gray_img = color.rgb2gray(img)
            else:
                gray_img = img  # Zakładamy, że jest już w skali szarości

            # Przeskalowanie wartości pikseli
            gray_img = img_as_ubyte(gray_img)
            gray_img //= 4  # Redukcja do 5 bitów na piksel (64 poziomy)

            # Obliczanie GLCM i cech tekstur
            glcm = graycomatrix(gray_img, distances, angles, 64, symmetric=True, normed=True)
            features = {'file': file_name,
                        'category': file_name.split('_')[0]}  # Użyj części nazwy pliku jako kategorii

            for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
                for dist in distances:
                    for ang in angles:
                        features[f'{prop}_d{dist}_a{int(np.degrees(ang))}'] = graycoprops(glcm, prop)[
                            distances.index(dist), angles.index(ang)]

            features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"Wyekstrahowano cechy dla {len(features_df)} próbek.")
    return features_df


def classify_features(features_df):
    if features_df.empty or 'category' not in features_df:
        print("Brak danych do klasyfikacji lub brakująca kategoria 'category'.")
        return

    print("Klasyfikacja cech...")
    X = features_df.drop(['file', 'category'], axis=1)
    y = features_df['category']

    if len(set(y)) < 2:
        print("Niewystarczająca liczba klas do klasyfikacji.")
        return

    # Podział danych na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Utworzenie i trening klasyfikatora k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Obliczenie i wyświetlenie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {accuracy:.4f}")


# Ścieżki do folderów źródłowych
source_folders = [

]

# Ścieżka do folderu docelowego
destination = r"D:\studia wojtek\2 stopien\semestr 1\Programowanie w obliczeniach inteligentnych\cropped"

# Wykonanie przycinania obrazów ze wszystkich folderów źródłowych
crop_and_save_images(source_folders, destination)

# Ekstrakcja cech tekstur z przyciętych obrazów i zapisanie ich do pliku CSV
features_df = extract_texture_features(destination)
features_df.to_csv('texture_features.csv', index=False)

# Klasyfikacja cech i wydrukowanie dokładności
classify_features(features_df)
