import pickle
import os

file_path = 'class_labels.pkl'

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found at {os.path.abspath(file_path)}")
else:
    try:
        with open(file_path, 'rb') as f:
            class_indices = pickle.load(f)

        idx_to_class_map = {v: k for k, v in class_indices.items()}

        print("Isi class_indices dari class_labels.pkl:")
        print(class_indices)
        print("\nIsi idx_to_class (terbalik):")
        print(idx_to_class_map)

        # Coba akses indeks 0
        if 0 in idx_to_class_map:
            print(f"\nKelas untuk indeks 0 adalah: {idx_to_class_map[0]}")
        else:
            print("\nIndeks 0 TIDAK ditemukan di idx_to_class!")

    except Exception as e:
        print(f"Error saat memuat class_labels.pkl: {e}")