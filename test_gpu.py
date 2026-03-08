import os
import sys

# Dodajemy ścieżki do plików DLL Twojej karty bezpośrednio do działającego skryptu
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
cudnn_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp" # sprawdź czy masz ten folder

if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
    print("Zlokalizowano CUDA 12.1!")

import tensorflow as tf
print("--- WYNIK TESTU ---")
print("Urządzenia:", tf.config.list_physical_devices('GPU'))