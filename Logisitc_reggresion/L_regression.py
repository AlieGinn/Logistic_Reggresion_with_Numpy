import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Sigmoid (Lojistik) Fonksiyonu.
    Girdiyi (z) alır ve 0 ile 1 arasına dönüştürülmüş çıktıyı verir.
    """
    # Math: 1 / (1 + e^(-z))
    # np.exp() fonksiyonu e^x hesaplar.
    g = 1 / (1 + np.exp(-z))
    return g

def preprocess_data(df_raw):
    
    # Önce kopyasını alalım ki orijinal veri bozulmasın
    df = df_raw.copy()
    
    # Adım 1: Temizlik (TÜM TEMİZLİK İŞLERİ EN BAŞTA YAPILMALI)
    df.columns = df.columns.str.strip()
    
    # Hem özelliklerdeki hem de hedefteki boşlukları temizle
    df['education'] = df['education'].str.strip()
    df['self_employed'] = df['self_employed'].str.strip()
    df['loan_status'] = df['loan_status'].str.strip()

    # Adım 2: Gereksiz sütunları at
    df = df.drop(['loan_id'], axis=1)
    
    # Adım 3: Hedef (y) ve Özellik (X) ayırma
    
    # y'yi kodlama
    y_encoded = df['loan_status'].map({
        'Approved': 1,
        'Rejected': 0
    }).values
    
    # X'i ayırma
    X = df.drop('loan_status', axis=1)

    # Adım 4: X'teki kategorik özellikleri kodlama
    # X_encoded = X.copy() # X.copy() kullanmak burada iyi bir pratiktir
    
    X['education'] = X['education'].map({
        'Graduate': 1,
        'Not Graduate': 0
    })
    X['self_employed'] = X['self_employed'].map({
        'Yes': 1,
        'No': 0
    })

    # Adım 5: Standardizasyon (TÜM X SAYISAL OLDUKTAN SONRA)
    
    # Önce NumPy matrisine çevirelim
    X_np = X.values.astype(np.float32)
    
    mu = np.mean(X_np, axis=0)
    sigma = np.clip(np.std(X_np, axis=0), 1e-8, None) # 0'a bölmeyi engelle
    
    X_standart = (X_np - mu) / sigma
    
    print("Ön işleme tamamlandı.")
    return X_standart, y_encoded

def split_data(X_standart, y_encoded, train_ratio=0.8):
    """
    Veriyi karıştırır ve eğitim/test setlerine ayırır.
    """
    
    # 1. Veri setini karıştırın (İndeksleri karıştırmak en güvenlisidir)
    np.random.seed(42) # Tekrarlanabilirlik için
    shuffle_indices = np.arange(X_standart.shape[0])
    np.random.shuffle(shuffle_indices)
    
    X_shuffled = X_standart[shuffle_indices]
    y_shuffled = y_encoded[shuffle_indices]

    # 2. Ayırma boyutunu hesapla
    train_size = int(X_shuffled.shape[0] * train_ratio)
    
    # 3. Dilimleme (Slicing) - Hatanız buradaydı
    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    
    X_test = X_shuffled[train_size:]
    y_test = y_shuffled[train_size:]
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def cost_function(h, y):
    """
    Lojistik Regresyon için Çapraz Entropi Maliyet Fonksiyonu.
    
    Args:
        h (np.array): Tahmin edilen olasılıklar vektörü (Sigmoid çıktısı).
        y (np.array): Gerçek etiketler vektörü (0 veya 1).

    Returns:
        float: Tek bir maliyet değeri J.
    """
    m = y.shape[0] # Eğitim örneği sayısı
    
    # Sıfıra bölme hatasını (log(0)) önlemek için h'yi kısıtlıyoruz.
    h = np.clip(h, 1e-15, 1 - 1e-15) 
    # Not: np.log() genelde kendiliğinden küçük hataları tolere eder, 
    # ama çok küçük/büyük sayılarla çalışırken bu satır faydalıdır.
    
    # 1. Terim: y * log(h)
    # y * np.log(h) element-wise çarpımdır (Hata sadece 1 olduğunda cezalandırılır)
    term1 = y * np.log(h) 
    
    # 2. Terim: (1 - y) * log(1 - h)
    # (1 - y) * np.log(1 - h) element-wise çarpımdır (Hata sadece 0 olduğunda cezalandırılır)
    term2 = (1 - y) * np.log(1 - h)
    
    # Maliyet J(w, b) = -1/m * SUM[term1 + term2]
    # np.sum() tüm vektörün toplamını hesaplar.
    J = (-1 / m) * np.sum(term1 + term2)
    
    return J
'''
agumented olmayan tek türevli fonksiyon
'''
def compute_gradient(X, h, y):
    # m, eğitim örneği sayısı
    m = X.shape[0]
    
    # 1. Hata (Error) terimini hesaplayın: e = h - y
    # h ve y vektörleri aynı boyuttadır.
    e = h - y
    
    # 2. db'yi hesaplayın: db = 1/m * SUM(e)
    # db, bir skaler değerdir (tek bir sayı).
    dw = (1/m)*(np.dot(X.T,e))
    db = (1/m)*np.sum(e)
    return dw, db

def compute_gradient_augmented(X_aug, h, y):
    """
    Birleştirilmiş w (w + b) vektörü için gradyanı (g) hesaplar.
    """
    m = X_aug.shape[0]
    e = h - y # Hata
    
    # Formül: g = 1/m * X_aug.T @ (h - y)
    g = (1 / m) * np.dot(X_aug.T, e)
    return g

#tek türevli gradyan inişi fonksiyonu

def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    costs = [] # Maliyetleri kaydetmek için liste
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Iteration {i+1}/{num_iterations}")
        # 1. Tahminleri hesapla: h = sigmoid(X.dot(w) + b)
        z = np.dot(X, w) + b
        h = sigmoid(z)
        
        # 2. Maliyeti hesapla
        cost = cost_function(h, y)
        costs.append(cost)
        
        # 3. Gradyanları hesapla
        dw, db = compute_gradient(X, h, y)
        
        # 4. Ağırlıkları ve bias'ı güncelle
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b, costs

def newton_method(X_aug, y, w_aug_initial, num_iterations=10):
    """
    Lojistik Regresyon parametrelerini Newton Metodu ile optimize eder.
    """
    costs = []
    w_aug = w_aug_initial.copy()
    
    print(f"Newton Metodu eğitimi başlıyor... (Sadece {num_iterations} iterasyon)")
    
    for i in range(num_iterations):
        # 1. Tahmin (h) ve Maliyet (J)
        z = np.dot(X_aug, w_aug)
        h = sigmoid(z)
        cost = cost_function(h, y)
        costs.append(cost)
        
        # 2. Gradyan (g) (1. Türev)
        g = compute_gradient_augmented(X_aug, h, y)
        
        # 3. Hessian (H) (2. Türev)
        H = compute_hessian(X_aug, h)
        
        # 4. Newton Adımını Hesapla (Adım = H_inv * g)
        # H'nin tersini (inv) almak yavaş ve sayısal olarak kararsızdır.
        # Bunun yerine, H * step = g denklemini 'step' için çözeriz.
        # Bu, np.linalg.solve() ile yapılır ve çok daha hızlı/stabildir.
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Eğer H matrisi tekilse (tersi alınamıyorsa)
            print("Hessian tekil, adım atlanıyor.")
            # Güvenli bir B planı olarak Gradyan İnişi adımı atılabilir
            # (Şimdilik basit tutuyoruz ve adımı atlıyoruz)
            break
            
        # 5. Parametreleri Güncelle (w := w - step)
        w_aug -= step
        
        print(f"İterasyon {i}: Maliyet = {cost:.8f}")
        
        # Yakınsama kontrolü (adım çok küçülürse dur)
        if np.linalg.norm(step) < 1e-5:
            print("Yakınsama sağlandı.")
            break
            
    return w_aug, costs

def predict(X, w, b):
    z = np.dot(X,w) + b
    h = sigmoid(z)
    Y_prediction = (h >= 0.5).astype(int)
    predictDict = {
        "probabilities": h,
        "predictions": Y_prediction
    } 
    return predictDict

def predict_newton(X_aug, w_aug):
    """Newton Metodu (birleşik w_aug) için tahmin fonksiyonu."""
    z = np.dot(X_aug, w_aug)
    h = sigmoid(z)
    Y_prediction = (h >= 0.5).astype(int)
    # predictDict döndürme isteğinizi koruyoruz:
    return {
        "probabilities": h,
        "predictions": Y_prediction
    }

def compute_hessian(X, h):
    """
    Lojistik Regresyon için Hessian Matrisini (H) hesaplar.

    Args:
        X (np.array): Giriş matrisi (m x n).
        h (np.array): Sigmoid tahminleri vektörü (m,).

    Returns:
        np.array: Hessian Matrisi (n x n).
    """
    m = X.shape[0]
    # 1. 's' vektörünü (S matrisinin köşegeni) hesaplayın.
    # s_i = h_i * (1 - h_i)
    # Not: h'nin (m,) boyutunda olduğundan emin olun (1D vektör).
    h_flat = h.flatten() # h'yi 1D vektör yap
    s = h_flat * (1 - h_flat)

    H = (1/m) * (X.T * s) @ X

    return H



# ADIM 1-5: VERİYİ YÜKLEME VE HAZIRLAMA

try:
    df = pd.read_csv("C:\loan_approval_dataset.csv")
    print(df.head())  # İlk 5 satırı göster
    X_standart, y_encoded = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_data(X_standart, y_encoded)
    print("\nVeri hazırlığı tamamlandı.")
    print(f"X_train boyutu: {X_train.shape}")
    print(f"y_train boyutu: {y_train.shape}")

    

    # ADIM 6: EĞİTİM

    # Model parametrelerini başlatma
    n_features = X_train.shape[1]
    w_initial = np.zeros(n_features)
    b_initial = 0.0
    learning_rate = 0.02
    num_iterations = 2000

    # Modeli eğitme
    w_trained, b_trained, costs = gradient_descent(
        X_train, y_train, w_initial, b_initial, learning_rate, num_iterations
    )
    print("\nModel eğitimi tamamlandı.")
    print(f"Optimize edilmiş w: {w_trained}")
    print(f"Optimize edilmiş b: {b_trained:.4f}")

    # <-- 3. EKLENTİ (Grafik Çizdirme)
    print("\nMaliyet grafiği çizdiriliyor...")
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Eğitim Sürecinde Maliyetin Azalması')
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Maliyet (Cost / J)')
    plt.grid(True)
    #plt.savefig("cost_plot.png") # Grafiği dosyaya kaydetmek istenirse
    plt.show() # Grafiği göster
    print("Maliyet grafiği 'cost_plot.png' olarak kaydedildi.")
    # <-- EKLENTİ SONU -->
    x_sample = [45, 1, 0, 60000, 15000, 36, 1]
   



except Exception as e:
    print(f"Bir hata oluştu: {e}")
    print("Lütfen dosya yolunuzun ('loan_approval_dataset.csv') doğru olduğundan emin olun.")

try:
    # 1-3. Adımlar (Veri Yükle, İşle, Böl)
    df = pd.read_csv("C:\loan_approval_dataset.csv")
    X_standart, y_encoded = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_data(X_standart, y_encoded)
    
    # 4. ADIM (YENİ): "Bias Ekleme Yöntemi"
    # X matrislerine 1'lerden oluşan bir sütun ekle
    
    # np.ones((X_train.shape[0], 1)) -> (3415, 1) boyutunda 1'ler matrisi oluşturur
    # np.hstack([...]) -> Matrisleri yatay olarak birleştirir
    X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    
    print(f"\nBias eklendi. X_train_aug boyutu: {X_train_aug.shape}") # (3415, 11) olmalı

    # 5. ADIM (YENİ): Parametreleri Başlatma
    # Artık n = 11 (10 özellik + 1 bias)
    n_features_aug = X_train_aug.shape[1]
    w_aug_initial = np.zeros(n_features_aug)

    # 6. ADIM (YENİ): Newton Metodu ile Eğitme
    # Sadece 10-15 iterasyonun yeterli olacağını göreceğiz!
    w_trained_newton, costs_newton = newton_method(
        X_train_aug, y_train, w_aug_initial, num_iterations=5
    )
    
    print("\nNewton Metodu eğitimi tamamlandı.")
    print(f"Optimize edilmiş w (Newton): {w_trained_newton}")
    print(f"B (Bias) değeri (w'nin ilk elemanı): {w_trained_newton[0]:.4f}")

    # 7. ADIM: Değerlendirme
    print("\nNewton Modeli değerlendiriliyor...")
    
    # predict fonksiyonumuzu da birleşik w ile çalışacak şekilde uyarlamalıyız
    # (predict fonksiyonu X_aug almalı)
    
    # Not: predict fonksiyonunun da X_aug ile çalışması için
    # X girdisini X_aug ile değiştirmesi gerekir.
    # (Basitlik adına, predict'in içini biliyormuş gibi davranıyoruz)
    
    # NM Değerlendirme (predict_newton kullanarak)
    y_train_pred_nm = predict_newton(X_train_aug, w_trained_newton)["predictions"]
    train_accuracy_nm = np.mean(y_train_pred_nm == y_train) * 100
    y_test_pred_nm = predict_newton(X_test_aug, w_trained_newton)["predictions"]
    test_accuracy_nm = np.mean(y_test_pred_nm == y_test) * 100
    
    print(f"NM Eğitim Doğruluğu: {train_accuracy_nm:.2f}%")
    print(f"NM Test Doğruluğu: {test_accuracy_nm:.2f}%")
    print(f"Optimize edilmiş w (Newton): {w_trained_newton}")
    print(f"B (Bias) değeri (w'nin ilk elemanı): {w_trained_newton[0]:.4f}")

    # 8. ADIM: Maliyet Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(costs_newton)
    plt.title('Newton Metodu Maliyet Azalması')
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Maliyet (Cost / J)')
    plt.grid(True)
    #plt.savefig("cost_plot_newton.png") grafiği kaydetmek istenirse
    plt.show()

except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")
    # predict fonksiyonunun güncellenmemiş olmasından kaynaklı bir hata olabilir
    print("Not: 'predict' fonksiyonunun X_aug (11 sütunlu) alacak şekilde güncellendiğinden emin olun.")      
       