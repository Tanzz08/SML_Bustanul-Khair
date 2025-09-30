import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump

def preprocess_paysim(data, target_column, save_path, apply_undersample=False):
    # 1. Drop kolom yang tidak relevan
    if "isFlaggedFraud" in data.columns:
        data = data.drop(["isFlaggedFraud"], axis=1)

    # 2. Feature engineering: sender_type & receiver_type
    data["sender_type"] = data["nameOrig"].str[0]
    data["receiver_type"] = data["nameDest"].str[0]

    # 3. Drop kolom ID
    data = data.drop(["nameOrig", "nameDest"], axis=1)

    # 4. Tambah fitur errorBalance
    data["errorBalanceOrig"] = data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]
    data["errorBalanceDest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]

    # 5. Tentukan fitur numerik & kategorikal
    numeric_features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = ["type", "sender_type", "receiver_type"]

    # pastikan target tidak ikut
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # 6. Pipeline numerik
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # 7. Pipeline kategorikal
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 8. Gabungkan dengan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 9. Pisahkan X dan y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 10. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 11. Fit-transform train, transform test
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 12. Ambil nama kolom hasil transformasi
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features))
    )

    # 13. Bungkus jadi DataFrame
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train, name=target_column)
    y_test = pd.Series(y_test, name=target_column)

    # 14. Terapkan Undersampling (opsional)
    if apply_undersample:
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print("✅ Random Undersampling diterapkan: Data training diperkecil & balance.")

    # 15. Simpan pipeline
    dump(preprocessor, save_path)
    print(f"✅ Pipeline preprocessing berhasil disimpan di {save_path}")

    return X_train, X_test, y_train, y_test
