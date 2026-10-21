# -*- coding: utf-8 -*-
"""
ZAMAN SERİSİ ANALİZİ - 5. HAFTA (15.10.2026) ÖDEVİ: ŞEKİL 1.13
Standart ve En Sade Spyder Uygulama Kodu (statsmodels ile)
(Her bölümü Spyder'da "Shift + Enter" ile tek tek çalıştırabilirsin)
"""

# %% 1. ADIM: KİTAPTAKİ ŞEKİL 1.13 VERİSİNİ ÜRETME VE ZAMAN SERİSİ GRAFİĞİ
import numpy as np
import pandas as pd
from numpy.random import normal
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# Kitaptaki Şekil 1.13'ün birebir kodu (Sayfa 25):
np.random.seed(1357)                        # Her çalıştırmada kitaptakiyle aynı grafik çıksın diye sabitliyoruz
a = normal(size=300)                        # 300 adet standart normal akgürültü (şok) verisi üretir
b = normal(size=300)
c = normal(size=300)

x = np.cumsum(a)                            # Kümülatif (birikimli) toplam alarak Rastgele Yürüyüş yapar
y = np.cumsum(b)
z = np.cumsum(c)

xyz = pd.DataFrame({'x': x, 'y': y, 'z': z})  # x, y, z serilerini tabloya (DataFrame) koyar
xyz.index = range(1, 301)                     # Zaman indeksini 1'den 300'e kadar numaralandırır

print("--- Orijinal Serilerin İlk 10 Gözlemi ---")
print(xyz.head(10).round(3))

# 1) Zaman Serisi Grafiği Çizimi (Kitaptaki Şekil 1.13):
xyz.plot(style=['-', '--', ':'])
plt.title('Sekil 1.13: Zaman Serisi Grafigi (Rastgele Yuruyus)')
plt.xlabel('Zaman (t)')
plt.ylabel('Seri Degerleri')
plt.show()


# %% 2. ADIM: LAG (GECİKME) İŞLEMCİSİ (B)
# .shift(1) seriyi 1 dönem, .shift(2) ise 2 dönem ileri (aşağı/sağa) kaydırır:
x_lag1 = xyz['x'].shift(1)
x_lag2 = xyz['x'].shift(2)

lag_tablosu = pd.DataFrame({
    'x (Orijinal)': xyz['x'],
    'Lag 1 (x_t-1)': x_lag1,
    'Lag 2 (x_t-2)': x_lag2
})

print("\n--- Lag (Gecikme) Tablosu (İlk 10 Gözlem) ---")
print(lag_tablosu.head(10).round(3))

# Kaymayı grafikte net görmek için ilk 30 gözlemi çizdiriyoruz:
lag_tablosu.head(30).plot(style=['-', '--', ':'])
plt.title('x Serisi ve Gecikmeleri: Lag 1 ve Lag 2 (Ilk 30 Gozlem)')
plt.xlabel('Zaman (t)')
plt.ylabel('Degerler')
plt.show()


# %% 3. ADIM: FARK İŞLEMCİSİ (DIFFERENCE)
# .diff(1) komutu her değerden bir önceki değeri çıkarır (Delta x_t = x_t - x_{t-1}):
fark_xyz = xyz.diff(1)
fark_xyz = fark_xyz.dropna()                # İlk satırdaki boş (NaN) gözlemi siler

print("\n--- 1. Dereceden Fark Serisi (İlk 10 Gözlem) ---")
print(fark_xyz.head(10).round(3))

# 1. Farkı alınmış serinin zaman serisi grafiği:
fark_xyz['z'].plot()
plt.axhline(0, color='red', linestyle='--')
plt.title('x Serisinin 1. Dereceden Fark Grafigi (Duraganlastirilmis Seri)')
plt.xlabel('Zaman (t)')
plt.ylabel('1. Fark (Delta x)')
plt.show()


# %% 4. ADIM: ACF GRAFİKLERİ, VERİLERİ VE DURAĞANLIK İNCELEMESİ
# plot_acf() fonksiyonu hem çubukları hem de mavi gölgeli %95 güven sınırını otomatik çizer.
# acf() fonksiyonu ise gecikmelerin sayısal otokorelasyon değerlerini verir.

# A) Orijinal x serisinin ACF Grafiği ve Değerleri (Durağan Değil):
plot_acf(xyz['x'], lags=20)
plt.title('Orijinal x Serisi ACF Grafigi (Yavas Azaliyor -> Duragan Degil)')
plt.xlabel('Gecikme Sayisi (Lag)')
plt.ylabel('ACF')
plt.show()

acf_x = acf(xyz['x'], nlags=10)
print("\n--- Orijinal x Serisi ACF Değerleri (Lag 1'den 10'a kadar) ---")
print(np.round(acf_x[1:], 3))


# B) 1. Farkı alınmış x serisinin ACF Grafiği ve Değerleri (Durağan / Akgürültü):
plot_acf(fark_xyz['x'], lags=20)
plt.title('1. Fark (Delta x) Serisi ACF Grafigi (Mavi Sinir Icinde -> Duragan)')
plt.xlabel('Gecikme Sayisi (Lag)')
plt.ylabel('ACF')
plt.show()

acf_fark_x = acf(fark_xyz['x'], nlags=10)
print("\n--- 1. Fark (Delta x) Serisi ACF Değerleri (Lag 1'den 10'a kadar) ---")
print(np.round(acf_fark_x[1:], 3))