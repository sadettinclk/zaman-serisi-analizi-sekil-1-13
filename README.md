# Python ile Zaman Serileri Analizi Dersi Ödevi — Şekil 1.13

 
**Konu:** Standart Normal Rastgele Yürüyüş (*Random Walk*) Serisinin Üretilmesi, Gecikme İşlemcisi ($B$), Fark İşlemcisi ($\Delta$), ACF Grafikleri ve Durağanlık İncelemesi

## 1. Veri Üretimi ve Orijinal Zaman Serisi Grafiği (Şekil 1.13)

Bu çalışmada, *Applied Time Series Analysis and Forecasting with Python* (Huang & Petukhina, 2022, s. 24–25, Örnek 1.8) kapsamında yer alan **Şekil 1.13 Standart Normal Rastgele Yürüyüş (*Random Walk*)** zaman serisi simüle edilerek durağanlık yapısı analiz edilmiştir. Teorik olarak $\varepsilon_1, \varepsilon_2, \dots, \varepsilon_T$ ortalaması $0$ ve varyansı $\sigma^2 = 1$ olan bağımsız ve aynı dağılımlı ($\varepsilon_t \sim N(0,1)$) standart normal akgürültü (*white noise*) serisi olmak üzere, rastgele yürüyüş süreci birikimli toplam formunda $X_t = X_{t-1} + \varepsilon_t = \sum_{i=1}^{t} \varepsilon_i$ ($t = 1, \dots, 300$) eşitliği ile tanımlanmaktadır.

```python
import numpy as np
import pandas as pd
from numpy.random import normal
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

np.random.seed(1357)
a = normal(size=300)
b = normal(size=300)
c = normal(size=300)

x = np.cumsum(a)
y = np.cumsum(b)
z = np.cumsum(c)

xyz = pd.DataFrame({'x': x, 'y': y, 'z': z})
xyz.index = range(1, 301)

print("--- Orijinal Serilerin İlk 10 Gözlemi ---")
print(xyz.head(10).round(3))

xyz.plot(style=['-', '--', ':'])
plt.title('Sekil 1.13: Zaman Serisi Grafigi')
plt.xlabel('Zaman (t)')
plt.ylabel('Seri Degerleri')
plt.show()
```

### Tablo 1: Orijinal Serilerin İlk 10 Gözlemi

| Zaman ($t$) | `x` Serisi | `y` Serisi | `z` Serisi |
| :---: | :---: | :---: | :---: |
| **1** | 0.425 | 1.090 | 0.537 |
| **2** | -0.188 | 1.077 | 0.444 |
| **3** | 0.017 | 1.499 | 0.520 |
| **4** | -0.034 | 0.942 | 0.543 |
| **5** | -0.160 | 0.348 | 1.386 |
| **6** | 2.910 | -1.228 | 1.609 |
| **7** | 2.389 | -2.162 | 2.485 |
| **8** | 3.188 | -2.415 | 2.111 |
| **9** | 3.435 | -1.281 | 1.828 |
| **10** | 3.282 | -1.040 | 3.841 |

[1_zaman_serisi.png]

### Grafik 1 Yorumu ve Durağanlık Değerlendirmesi
* **Zayıf Durağanlık Koşulları:** Bir zaman serisinin zayıf durağan olabilmesi için beklenen değerinin sabit ($E(X_t) = \mu$), varyansının sabit ($\text{Var}(X_t) = \sigma^2$) ve otokovaryansının zamandan bağımsız olarak yalnızca gecikme sayısına bağlı ($\text{Cov}(X_t, X_{t+k}) = \gamma_k$) olması gerekmektedir.
* **Ortalama ve Varyans Yapısı:** Grafik incelendiğinde `x`, `y` ve `z` serilerinin belirli bir sabit ortalama etrafında salınmadığı ve zaman içinde büyük saçılımlar gösterdiği görülmektedir. Düz çizgiyle gösterilen `x` serisi ve noktalı çizgiyle gösterilen `z` serisi $+20$ seviyelerine kadar yükselen bir stokastik eğilim sergilerken, kesikli çizgiyle gösterilen `y` serisi $-17$ seviyesine kadar düşmektedir.
* **Teorik Gerekçe ve Sonuç:** Rastgele yürüyüş sürecinde beklenen değer $E(X_t) = 0$ olsa da varyans $\text{Var}(X_t) = t\sigma^2$ ve otokovaryans $\gamma_X(t, s) = \min(t, s)\sigma^2$ zamana ($t$) bağlıdır. Varyansı zamana bağlı olarak büyüdüğü ve ortalaması sabit kalmadığı için **orijinal seriler durağan değildir**.

## 2. Gecikme (Lag) İşlemcisi ($B$) Uygulaması

$B$ gecikme işlemcisi olmak üzere, $x_t$ serisinin $k$ dönem kaydırılmasıyla elde edilen gecikmeli seri $B^k x_t = x_{t-k}$ şeklinde gösterilir. Uygulamada `x` serisine `.shift(1)` ve `.shift(2)` fonksiyonları uygulanarak 1 dönem gecikmeli ($x_{t-1}$) ve 2 dönem gecikmeli ($x_{t-2}$) seriler oluşturulmuştur.

```python
x_lag1 = xyz['x'].shift(1)
x_lag2 = xyz['x'].shift(2)

lag_tablosu = pd.DataFrame({
    'x': xyz['x'],
    'Lag 1 (x_t-1)': x_lag1,
    'Lag 2 (x_t-2)': x_lag2
})

print("--- Lag Tablosu (İlk 10 Gözlem) ---")
print(lag_tablosu.head(10).round(3))

lag_tablosu.head(30).plot(style=['-', '--', ':'])
plt.title('x Serisi ve Gecikmeleri: Lag 1 ve Lag 2')
plt.xlabel('Zaman (t)')
plt.ylabel('Degerler')
plt.show()
```

### Tablo 2: `x` Serisi ve Gecikmelerinin İlk 10 Gözlemi

| Zaman ($t$) | `x` (Orijinal $x_t$) | `Lag 1` ($x_{t-1}$) | `Lag 2` ($x_{t-2}$) |
| :---: | :---: | :---: | :---: |
| **1** | 0.425 | NaN | NaN |
| **2** | -0.188 | 0.425 | NaN |
| **3** | 0.017 | -0.188 | 0.425 |
| **4** | -0.034 | 0.017 | -0.188 |
| **5** | -0.160 | -0.034 | 0.017 |
| **6** | 2.910 | -0.160 | -0.034 |
| **7** | 2.389 | 2.910 | -0.160 |
| **8** | 3.188 | 2.389 | 2.910 |
| **9** | 3.435 | 3.188 | 2.389 |
| **10** | 3.282 | 3.435 | 3.188 |

[2_lag_grafigi.png]

### Grafik 2 ve Tablo Yorumu
* $t = 1$ başlangıç noktasında $t = 0$ gözlemi bulunmadığından `Lag 1` serisinin ilk satırı, `Lag 2` serisinin ise ilk iki satırı boş (`NaN`) olarak oluşmuştur. Orijinal serinin 1. dönemdeki `0.425` değeri, `Lag 1`'de 2. döneme, `Lag 2`'de ise 3. döneme taşınmıştır.
* Grafikte kesikli çizgi (`Lag 1`) ve noktalı çizgi (`Lag 2`), düz çizgiyle gösterilen orijinal `x` serisinin zaman ekseninde 1 ve 2 birim sağa ötelenmiş halidir. Gecikme işlemcisi serinin değerlerini veya varyansını değiştirmediğinden durağanlık yapısını etkilemez.

## 3. Birinci Dereceden Fark İşlemcisi ($\Delta$) Uygulaması

Zaman serisinin son değerlerinden bir dönem önceki değerlerinin çıkarılması işlemine **1. dereceden fark işlemi** denir ve $\Delta x_t = x_t - x_{t-1} = (1 - B)x_t$ eşitliğiyle hesaplanır. Rastgele yürüyüş serisinde $x_t - x_{t-1} = a_t$ olduğundan, 1. fark alındığında birikimli stokastik trend elenmekte ve geriye standart normal akgürültü şokları kalmaktadır.

```python
fark_xyz = xyz.diff(1)
fark_xyz = fark_xyz.dropna()

print("--- 1. Dereceden Fark Serisi (İlk 10 Gözlem) ---")
print(fark_xyz.head(10).round(3))

fark_xyz['x'].plot()
plt.axhline(0, color='red', linestyle='--')
plt.title('x Serisinin 1. Dereceden Fark Grafigi')
plt.xlabel('Zaman (t)')
plt.ylabel('1. Fark (Delta x)')
plt.show()
```

### Tablo 3: 1. Dereceden Fark Serisinin İlk 10 Gözlemi

| Zaman ($t$) | $\Delta x_t = x_t - x_{t-1}$ | $\Delta y_t = y_t - y_{t-1}$ | $\Delta z_t = z_t - z_{t-1}$ |
| :---: | :---: | :---: | :---: |
| **2** | -0.613 | -0.013 | -0.093 |
| **3** | 0.206 | 0.422 | 0.077 |
| **4** | -0.051 | -0.557 | 0.022 |
| **5** | -0.127 | -0.593 | 0.843 |
| **6** | 3.070 | -1.577 | 0.224 |
| **7** | -0.521 | -0.934 | 0.875 |
| **8** | 0.800 | -0.253 | -0.373 |
| **9** | 0.247 | 1.134 | -0.283 |
| **10** | -0.153 | 0.240 | 2.013 |
| **11** | -0.224 | 0.020 | 0.565 |

[3_fark_grafigi.png]

### Grafik 3 Yorumu (Fark Durağanlığı)
* Orijinal `x` serisindeki yukarı yönlü stokastik eğilim 1. fark işlemi sonrasında tamamen ortadan kalkmıştır.
* Fark serisi, kırmızı kesikli referans çizgisiyle gösterilen **belirli bir ortalama ($0$ ortalaması) etrafında yaklaşık sabit sıçramalı (sabit saçılımlı, $\text{Var}(\Delta x_t) = 1$) bir varyans** göstermektedir. Yani seri **fark durağandır**.

## 4. ACF Grafikleri, Katsayıları ve Durağanlık Analizi

```python
plot_acf(xyz['x'], lags=20)
plt.title('Orijinal x Serisi ACF Grafigi')
plt.xlabel('Gecikme Sayisi (Lag)')
plt.ylabel('ACF')
plt.show()

acf_x = acf(xyz['x'], nlags=10)
print("--- Orijinal x Serisi ACF Değerleri (Lag 1-10) ---")
print(np.round(acf_x[1:], 3))

plot_acf(fark_xyz['x'], lags=20)
plt.title('1. Fark (Delta x) Serisi ACF Grafigi')
plt.xlabel('Gecikme Sayisi (Lag)')
plt.ylabel('ACF')
plt.show()

acf_fark_x = acf(fark_xyz['x'], nlags=10)
print("--- 1. Fark (Delta x) Serisi ACF Değerleri (Lag 1-10) ---")
print(np.round(acf_fark_x[1:], 3))
```

### Tablo 4: Orijinal `x` ve 1. Fark ($\Delta x$) Serisi ACF Katsayıları (Lag 1–10)

| Gecikme ($k$) | Lag 1 | Lag 2 | Lag 3 | Lag 4 | Lag 5 | Lag 6 | Lag 7 | Lag 8 | Lag 9 | Lag 10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Orijinal `x` ACF ($r_k$)** | 0.982 | 0.961 | 0.937 | 0.912 | 0.889 | 0.869 | 0.850 | 0.832 | 0.813 | 0.795 |
| **1. Fark ($\Delta x$) ACF ($r_k$)** | 0.086 | 0.122 | 0.049 | -0.085 | -0.002 | -0.071 | -0.021 | 0.061 | -0.026 | -0.056 |

[4_orijinal_acf.png]

* **Orijinal `x` Serisi ACF Yorumu:** Otokorelasyon katsayılarının güven sınırlarını (mavi alanı) aştığı ve oldukça yüksek değerlere sahip olduğu ($r_1 = 0.982$, $r_2 = 0.961$, $r_{10} = 0.795$) görülmektedir ($H_0: \rho_k = 0$ reddedilir). Gecikme sayısı arttıkça katsayılar doğrusal bir kalıpla çok yavaş azalmaktadır; bu yapı **orijinal `x` serisinin durağan olmadığını** kanıtlamaktadır.

[5_fark_acf.png]

* **1. Fark ($\Delta x$) Serisi ACF Yorumu:** Birinci farkı alınmış serinin ACF grafiğinde, **ilk gecikmeye (Lag 1) ait otokorelasyon katsayısı ($r_1 = 0.086$) $\pm 0.113$ mavi güven sınırlarının içerisinde** yer almaktadır. Sonraki gecikmelerde yalnızca 2. gecikme değeri ($r_2 = 0.122$) güven sınırını çok az aşmakta, diğer tüm gecikmeler güven sınırları içinde kalmaktadır. Akgürültü kriterine göre, **güven sınırını aşan değer kesinlikle ilk gecikmeye (Lag 1'e) ait olmadığı ve sayıca çok az (yalnızca 1 adet) olduğu için göz ardı edilebilir** ($H_0: \rho_k = 0$ kabul edilir). Dolayısıyla **1. fark serisi ($\Delta x_t$) yaklaşık Akgürültü (*White Noise*) serisidir ve durağandır**.
