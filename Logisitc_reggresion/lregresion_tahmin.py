import numpy as np
import pandas as pd
import L_regression

# 1. Örnek başvuru
sample_applicant = {
    "loan_id": 1001,
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 7500000,
    "loan_amount": 25000000,
    "loan_term": 12,
    "cibil_score": 720,
    "residential_assets_value": 5000000,
    "commercial_assets_value": 7000000,
    "luxury_assets_value": 10000000,
    "bank_asset_value": 4000000,
    "loan_status": "Approved"  # Gerçek durumu değil zaten prprocessedle eleniyor (sadece referans için)
}

# 2. DataFrame'e çevir
sample_df = pd.DataFrame([sample_applicant])

# 3. Ön işleme (sadece özellikleri kullan)
X_sample, _ = L_regression.preprocess_data(sample_df)

# 4. Eğitilmiş parametreleri yükle
# Bunları model eğitiminin çıktısından almalısın
# Örneğin:
w1 = L_regression.w_trained
b = L_regression.b_trained
X_train_augg = np.hstack((np.ones((X_sample.shape[0], 1)), X_sample))
w2 = L_regression.w_trained_newton

# 5. Tahmin yap
prediction1 = L_regression.predict(X_sample, w1, b)
prediction2 = L_regression.predict_newton(X_train_augg, w2)

print("Tahmin edilen olasılık:", prediction1["probabilities"])
print("Tahmin edilen sınıf (1=Approved, 0=Rejected): ", prediction1["predictions"])
print("Tahmin edilen olasılık:", prediction2["probabilities"])
print("Tahmin edilen sınıf (1=Approved, 0=Rejected):", prediction2["predictions"])
