Lojistik Regresyon: Sıfırdan Uygulama ve Optimizasyon Karşılaştırması

Bu proje, hazır makine öğrenimi kütüphaneleri (Scikit-learn vb.) kullanılmadan, Lojistik Regresyon algoritmasının matematiksel temellerini anlamak ve uygulamak amacıyla geliştirilmiştir.

Proje, kredi onayı tahminlemesi (loan_approval_dataset.csv) üzerinde çalışır ve iki farklı optimizasyon yöntemini karşılaştırır:

Gradyan İnişi (Gradient Descent - GD): Geleneksel, birinci türev tabanlı yöntem.

Newton Metodu (Newton's Method - NM): Hızlı yakınsayan, ikinci türev (Hessian) tabanlı yöntem.

Kod Yapısı ve Fonksiyonlar

Kodun ana bileşenleri şunlardır:

1. Temel Fonksiyonlar

sigmoid(z): Lineer çıktıyı 0-1 aralığına sıkıştıran aktivasyon fonksiyonu.

cost_function(h, y): Modelin başarısını ölçen Çapraz Entropi (Log-Loss) maliyet fonksiyonu. Sayısal kararlılık için np.clip kullanır.

2. Veri İşleme

preprocess_data(df):

Veriyi temizler (boşlukları siler).

Gereksiz sütunları (loan_id) atar (cibil_score model başarısı için tutulmuştur).

Kategorik verileri (Education, Self_Employed, Loan_Status) sayısal verilere (0/1) dönüştürür.

Standardizasyon (Z-Score): Tüm sayısal özellikleri aynı ölçeğe çeker.

split_data(...): Veriyi eğitim ve test setlerine ayırır.

3. Model 1: Gradyan İnişi (GD)

compute_gradient(...): Ağırlıkların türevini hesaplar.

gradient_descent(...): Belirlenen learning_rate ile iteratif olarak parametreleri günceller.

predict_gd(...): Ayrı w ve b parametreleri ile tahmin yapar.

4. Model 2: Newton Metodu (NM)

compute_hessian(...): Maliyet fonksiyonunun ikinci türev matrisini (Hessian) hesaplar.

newton_method(...): Hessian matrisini kullanarak minimum noktaya "sıçrayarak" çok hızlı ulaşır. Bias terimi ağırlık vektörüne dahil edilmiştir (Augmented Notation).

predict_newton(...): Birleşik ağırlık vektörü ile tahmin yapar.

Kurulum ve Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphanelerinin yüklü olması gerekir:

pip install numpy pandas matplotlib


Olası Hatalar ve Çözümleri (Troubleshooting)

Kodu çalıştırırken karşılaşabileceğiniz yaygın hatalar ve çözümleri aşağıdadır:

1. Hata: FileNotFoundError: [Errno 2] No such file or directory: 'loan_approval_dataset.csv'

Nedeni:
Python scripti, veri setini (csv dosyasını) çalıştığı klasörde bulamıyor. Genellikle dosya yolu tam verilmediğinde olur.

Çözüm:
Dosyanın tam yolunu (Absolute Path) verin.

Yanlış: pd.read_csv("loan_approval_dataset.csv")

Doğru (Windows): Dosya yolunun başına r koyarak (raw string) veya çift ters eğik çizgi \\ kullanarak tam yolu yazın.

# Yöntem 1 (Önerilen)
df = pd.read_csv(r"C:\Datas\regresyon_data\loan_approval_dataset.csv")

# Yöntem 2
df = pd.read_csv("C:\\Datas\\regresyon_data\\loan_approval_dataset.csv")


2. Hata: [Errno 13] Permission denied: 'cost_plot.png' (veya cost_comparison_with_cibil_plot.png)

Nedeni:
Kod, oluşturduğu grafiği bilgisayarınıza kaydetmeye çalışıyor (plt.savefig), ancak şunlardan biri oluyor:

Python'un o klasöre yazma izni yok (Örn: C:\Program Files içinde çalışıyorsanız).

Dosya (cost_plot.png) o sırada başka bir program tarafından açık tutuluyor.

Dosya yolu bir klasörü işaret ediyor ama dosya adı eksik.

Çözüm:

Yazılabilir Bir Klasör Seçin: Grafiği kaydetmek için tam yol belirtin. Örneğin masaüstüne veya belgelerim klasörüne kaydedin:

# Windows için örnek
plt.savefig(r"C:\Users\KullaniciAdiniz\Desktop\cost_plot.png")


Dosyayı Kapatın: Eğer png dosyası bir resim görüntüleyicide açıksa kapatıp tekrar deneyin.

Yönetici İzni: IDE'nizi (VS Code, PyCharm vb.) "Yönetici Olarak Çalıştır" diyerek açmayı deneyebilirsiniz (ama genellikle yolu değiştirmek daha güvenlidir).

3. Grafiklerin Gözükmemesi

Nedeni:
Bazı IDE'lerde veya sunucularda plt.show() grafiği ekrana basamayabilir veya kodun akışını kilitleyebilir.

Çözüm:
Kodumuzda plt.show() yerine plt.close() kullanılmıştır. Grafikler ekranda belirmeyebilir ancak .png dosyası olarak belirlediğiniz klasöre kaydedilir. Klasörü kontrol edin.

Sonuçlar

Bu çalışma sonucunda:

CIBIL Skoru olmadan model başarısı %63 civarında kalmıştır (Underfitting).

CIBIL Skoru eklendiğinde başarı %93 seviyelerine çıkmıştır.

Newton Metodu, Gradyan İnişi'ne göre çok daha az iterasyonda (yaklaşık 5-10 adımda) sonuca ulaşmıştır.
