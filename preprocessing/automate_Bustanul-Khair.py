import pandas as pd
import os
from preprocessing import preprocess_paysim

def automate_preprocessing(file_path, target_column="isFraud", apply_smote=False):
    """
    Fungsi untuk otomatis preprocessing dataset PaySim
    Args:
        file_path (str): path dataset CSV
        target_column (str): nama kolom target (default = isFraud)
        apply_smote (bool): apakah ingin menerapkan SMOTE di training data
    Returns:
        X_train, X_test, y_train, y_test : dataset siap latih
    """
    # 1. Load dataset
    print(f"📂 Loading dataset dari {file_path} ...")
    df = pd.read_csv(file_path)

    # 2. Panggil fungsi preprocessing dari preprocessing.py
    X_train, X_test, y_train, y_test = preprocess_paysim(
        df,
        target_column=target_column,
        save_path="preprocessor_pipeline.joblib",
        apply_smote=apply_smote
    )

    print("✅ Preprocessing selesai. Dataset siap dilatih 🚀")

    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    print("💾 Dataset berhasil disimpan di folder ")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "Paysim.csv")

    # Jalankan preprocessing + SMOTE
    X_train, X_test, y_train, y_test = automate_preprocessing(file_path, apply_smote=True)
