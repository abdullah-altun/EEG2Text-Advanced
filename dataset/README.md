# EEG2Text-Advanced Dataset

## Thought2Text 
Çalışma CVPR2017 EEG veri kümesini kullanıyor. Bu ham veriyi, üzerine GPT-4 tabanlı açıklamalar ekleyerek bir "EEG + Resim + Metin" üçlüsüne dönüştürüyor.

1. Orijinal Veri Kümesi: CVPR2017:
a) Katılımcı Sayısı ve Uyarıcılar
- Veri Kümesinde 6 farklı katılımcı yer alıyor.
- Her katılımcıya 40 farklı nesne kategorisinden seçilmiş 50 resim gösteriliyor. Toplamda bu, (6 katılımcı) x (50 resim) x (40 sınıf) biçiminde tanımlansa da pratikte 2000 civarinda örneğe karşılık geliyor

b) EEG Kayit Özellikleri
- Her katılımcı bir resmi gördüğünde, o anda 0.5 saniyelik EEG sinyali kaydediliyor.
- Kayıtlar, 128 kanallı bir EEG cihazından geliyor (yani 128 farklı elektrot verisi var).
- Örnekleme hızı (sampling rate) 1 kHz → 0.5 saniyede ≈ 500 örnek (sample) demek.
- Veri kümesinde, sıklıkla 55-95 Hz frekans aralığındaki banttan geçirilmiş (filtered) EEG versiyonu kullanılıyor, çünkü önceki çalışmalarda bu bant aralığının (yüksek gamma) belirli nesneleri ayırt etmede nispeten iyi sonuç verdiği görülmüş.
- Kaydın ilk 20 ms’lik kısmı (yaklaşık 20 örnek) “eski uyarıcının etkisi” (carry-over effect) olmasın diye atılıyor. Geriye kalan 480 ms civarı (yani ~440-460 örnek) standardize edilerek sabit uzunluklu bir matris hâline getiriliyor (128×440 boyutunda).

c) Verinin Bölünmesi
- Makalede, toplanan 2000 örnek (bazı ön-işlemlerle 7959 train, 1994 eval, 1987 test şeklinde görünüyor; her birinin “single segment” olarak kayda alınmış olmasıyla ilgili) ya da alt parçalara (block) bölünmesi söz konusu.
- Ana fikir: EEG sinyalleri eğitim (train), doğrulama (val) ve test olmak üzere ayrı setlere ayrılıyor. Böylece modelin hem öğrenme hem de genel performans testleri yapılabiliyor.

2. Final Veri Formatı: EEG + Resim + Caption
a) EEG (128×440 boyutunda sinyal)

- Belirtilen ön-işlemlerden geçirilmiş ham EEG verisi.

b) Görüntü (Image)

- 40 farklı nesne sınıfından birine ait resim.

c) Metin Açıklaması (Tek Cümlelik Caption)

- GPT-4 tarafından oluşturulmuş, insan onayından geçmiş tanımlama cümlesi.

d) (Ek Bilgi) Nesne Etiketi (Object Label)
- EEG sinyallerinde en baskın nesne sınıfı (ImageNet benzeri etiket).

- Örneğin “piano”, “car”, “mushroom” vb. 40 kategoriden biri.

- Bu kısım, Stage1 eğitiminde EEG ile CLIP imaj vektörlerinin hizalanması ve ek olarak, EEG üzerinden obje tahmini yapabilmek için de kullanılıyor.


1. block

- block_splits_by_image_all.pth ve block_splits_by_image_single.pth

Bu dosyalar, train/validation/test ayrımlarını (split) tutan Python obje dosyaları (PyTorch formatında).

- eeg_5_95_std.pth, eeg_14_70_std.pth, eeg_55_95_std.pth

Farklı bant aralıklarında (frequency band) filtrelenerek normalleştirilmiş EEG verilerini içeriyor.

- eeg_signals_raw_with_mean_std.pth

Ham EEG sinyalleri (henüz belirli bir bant filtre uygulamadan) ve/veya her kanalın ortalama (mean) ve standart sapma (std) bilgilerini içeriyor.

2. eeg_encoder_55-95_40_classes
- Bu klasörde, Stage1’de eğitilen EEG kodlayıcısının (ChannelNet tabanlı) çıktı model ağırlıkları (weights) ve konfigürasyonu bulunuyor:

3. images klasörü
- Thought2Text makalesinde, EEG’yi görsel özniteliklerle hizalarken (Stage1’de “EEG embeddings” ile “CLIP görsel gömlemleri”ni yakınlaştırma), renk veya detay gibi dikkat dağıtıcıları minimuma indirmek için resimleri “basitleştirme” (Gaussian blur + kenar (Canny) filtreleme) yaparak “sketch” adı verilen siyah-beyaz kenar görüntüleri oluşturuyorlar

