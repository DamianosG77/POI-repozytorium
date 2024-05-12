import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def crop_and_save_textures(input_dir, output_dir, crop_size, distances, angles):
    # Sprawdź czy katalog wyjściowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iteracja przez pliki w katalogu wejściowym
    for filename in os.listdir(input_dir):
        # Odczytaj obraz
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # Jeśli obraz nie może zostać wczytany, pomiń go
        if img is None:
            print(f"Nie można wczytać obrazu: {img_path}")
            continue

        # Pobierz wymiary obrazu
        height, width, _ = img.shape

        # Iteracja przez obrazy i wycinanie fragmentów tekstury
        for y in range(0, height - crop_size[1], crop_size[1]):
            for x in range(0, width - crop_size[0], crop_size[0]):
                # Wycięcie fragmentu tekstury
                texture = img[y:y + crop_size[1], x:x + crop_size[0]]

                # Utwórz katalog wyjściowy dla danej tekstury, jeśli nie istnieje
                output_texture_dir = os.path.join(output_dir, filename.split('.')[0])
                if not os.path.exists(output_texture_dir):
                    os.makedirs(output_texture_dir)

                # Zapisz wycięty fragment tekstury
                cv2.imwrite(os.path.join(output_texture_dir, f"{y}_{x}.jpg"), texture)

                # Oblicz cechy tekstury dla wyciętego fragmentu
                compute_and_save_texture_features(output_texture_dir, f"{y}_{x}.jpg", distances, angles)

    # Po przetworzeniu wszystkich tekstur, zapisz zbiór danych do pliku CSV
    save_dataset_to_csv(output_dir)

def compute_and_save_texture_features(input_dir, filename, distances, angles):
    # Odczytaj obraz
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Sprawdź czy obraz został wczytany
    if img is None:
        print(f"Nie można wczytać obrazu: {img_path}")
        return

    # Zmniejsz głębię jasności do 5 bitów (64 poziomy)
    img = (img / 64).astype(np.uint8)

    # Oblicz macierz zdarzeń szarości (GLCM)
    glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)

    # Oblicz cechy tekstury
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    correlation = graycoprops(glcm, 'correlation')
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    asm = graycoprops(glcm, 'ASM')

    # Zwróć cechy tekstury jako słownik
    texture_features = {
        'Filename': filename,
        'Distance': [],
        'Angle': [],
        'Dissimilarity': [],
        'Correlation': [],
        'Contrast': [],
        'Energy': [],
        'Homogeneity': [],
        'ASM': []
    }

    for d in range(len(distances)):
        for a in range(len(angles)):
            texture_features['Distance'].append(distances[d])
            texture_features['Angle'].append(angles[a])
            texture_features['Dissimilarity'].append(dissimilarity[d, a])
            texture_features['Correlation'].append(correlation[d, a])
            texture_features['Contrast'].append(contrast[d, a])
            texture_features['Energy'].append(energy[d, a])
            texture_features['Homogeneity'].append(homogeneity[d, a])
            texture_features['ASM'].append(asm[d, a])

    # Zapisz cechy do pliku
    output_file = os.path.join(input_dir, f"{filename.split('.')[0]}_features.txt")
    with open(output_file, 'w') as f:
        for key, value in texture_features.items():
            if isinstance(value, list):
                f.write(f"{key}: {', '.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")

    return texture_features

def save_dataset_to_csv(output_dir):
    # Utwórz listę cech wszystkich tekstur
    all_texture_features = []

    # Iteracja przez pliki w katalogu wyjściowym
    for subdir, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('_features.txt'):
                # Odczytaj cechy tekstury z pliku
                features_path = os.path.join(subdir, file)
                texture_features = {}
                with open(features_path, 'r') as f:
                    for line in f:
                        key, value = line.strip().split(': ')
                        if key == 'Filename':
                            texture_features[key] = value
                        else:
                            texture_features[key] = [float(x) for x in value.split(', ')]

                all_texture_features.append(texture_features)

    # Utwórz ramkę danych Pandas
    df = pd.DataFrame(all_texture_features)

    # Usuń plik CSV, jeśli już istnieje
    csv_file = os.path.join(output_dir, 'texture_dataset.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Zapisz ramkę danych do pliku CSV
    df.to_csv(csv_file, index=False)

def classify_texture_features(input_dir):
    # Znajdź plik CSV w katalogu wejściowym
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
    if not csv_files:
        print("Nie znaleziono pliku CSV w katalogu wejściowym.")
        return None
    elif len(csv_files) > 1:
        print("Znaleziono więcej niż jeden plik CSV w katalogu wejściowym. Wybierz jeden.")
        return None

    csv_file = os.path.join(input_dir, csv_files[0])

    # Wczytaj zbiór danych z pliku CSV
    df = pd.read_csv(csv_file)

    # Konwertuj kolumnę 'Filename' na unikalne identyfikatory numeryczne
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Filename'])

    # Usuń kolumnę 'Filename' przed przekazaniem danych do klasyfikatora
    X = df.drop(columns=['Filename', 'Label'])

    # Przekształć listy wartości w pojedyncze wartości liczbowe
    for column in X.columns:
        X[column] = X[column].apply(lambda x: np.mean(eval(x)))

    # Kolumna 'Label' będzie używana jako etykiety
    y = df['Label']

    # Podziel zbiór danych na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja klasyfikatora KNN
    knn = KNeighborsClassifier(n_neighbors=5)  # Wybierz liczbę sąsiadów

    # Uczenie klasyfikatora na danych treningowych
    knn.fit(X_train, y_train)

    # Testowanie klasyfikatora na danych testowych
    y_pred = knn.predict(X_test)

    # Obliczenie dokładności klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Ścieżki do katalogów wejściowego i wyjściowego oraz rozmiar wyciętych fragmentów
input_dir = r"C:\Users\USER\PycharmProjects\POI-repozytorium\Tekstury\tynk"
output_dir = r"C:\Users\USER\PycharmProjects\POI-repozytorium\Plik_wyjściowy"
crop_size = (128, 128)

# Definiowanie odległości pikseli i kierunków
distances = [1, 3, 5]
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Użyj funkcji do wycinania fragmentów tekstur i obliczania cech
crop_and_save_textures(input_dir, output_dir, crop_size, distances, angles)

# Klasyfikacja cech i uzyskanie dokładności klasyfikatora
accuracy = classify_texture_features(output_dir)

# Wyświetlenie uzyskanej dokładności klasyfikacji
if accuracy is not None:
    print("Dokładność klasyfikatora KNN:", accuracy)