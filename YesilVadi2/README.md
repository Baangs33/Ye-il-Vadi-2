# Yeşil Vadi 2.0 — FPS Hayatta Kalma, İnşaat ve Çiftlik

Python (Ursina / Panda3D) ile yazılmış, birinci şahıs bakışlı açık dünya oyunu. Hiçbir hazır model, doku ya da ses
dosyası kullanılmaz: arazi, ağaçlar, binalar, dokular (çim, toprak, kaya, ağaç kabuğu, tahta, taş duvar, kiremit...),
ikonlar ve sesler oyun açılırken kodla üretilir. Dokular ilk açılışta `onbellek/` klasörüne, sesler `sesler/` klasörüne
kaydedilir; sonraki açılışlar daha hızlı olur.

## Kurulum
Python 3.10+ gerekir.
```
pip install -r requirements.txt
python main.py
```
Windows'ta `baslat.bat` dosyasına çift tıklaman yeterli. İlk açılış 20-40 saniye sürebilir.

## Kontroller
| Tuş | İşlev |
|---|---|
| W A S D | Yürü |
| Shift | Koş (enerji harcar) |
| Boşluk | Zıpla |
| Fare | Etrafa bak |
| Sol Tık | Aleti kullan (kes, kaz, çapala, sula, ek, hasat et) |
| E | Etkileşim: plana malzeme ekle (basılı tut), böğürtlen topla, hayvan ürünü al, uyu, su doldur, tüccar, arabaya bin/in |
| 1-6 / Tekerlek | Alet seç: Balta, Kazma, Çapa, Sulama Kabı, Tohum, Çekiç |
| R | Tohum türünü değiştir / planı döndür |
| B | İnşaat Defteri |
| Tab | Envanter (yiyeceklere tıklayıp ye) |
| F | Hızlıca bir şey ye |
| X | Bakılan yarım planı kaldır (malzeme iade) |
| V | Araba kamerasını değiştir |
| Esc | Menü / pencereyi kapat |
| F5 | Hızlı kayıt |

## Oynanış
- **Kaynak topla:** Balta ile ağaç kes (kütük), kazma ile kaya (taş) ve paslı demir damarlarını (demir) kaz. Böğürtlen çalılarından yiyecek topla.
- **İnşa et (The Forest tarzı):** B ile defteri aç, bir yapı seç, yere yerleştir (yeşil = uygun). Plan hayalet olarak durur; plana bakıp E'yi basılı tuttukça kütükler / taşlar tek tek yerine oturur.
- **Yapılar:** Çadır, Kütük Kulübe, Büyük Kütük Ev, Taş Ev, Çiftlik Evi, Yatak, Tarla Parseli, Kümes, Ahır, Kuyu, Çit, Pazar Tezgâhı, Kamp Ateşi, Fener Direği, Taş Yol, Garaj.
- **Çiftçilik:** Tarla parselinde çapala → ek → sula. Islak toprakta ekin büyür; yağmur tarlayı kendiliğinden sular. Olgun ekine tıklayarak hasat et.
- **Hayvanlar:** Kümes / ahır kurduktan sonra tüccardan tavuk, koyun, inek al. Hazır olduklarında yumurta, yün, süt toplarsın.
- **Hayatta kalma:** Açlık ve enerji çubukları var. Koşmak, zıplamak ve alet sallamak enerji harcar. Ye, dinlen; akşamları yatakta / çadırda uyuyarak sabaha geç.
- **Kamp ateşi:** 2 sebzeden doyurucu çorba pişir.
- **Araba:** Ya tüccardan 2400 altına satın al, ya da Garaj inşa edip içindeki tezgâhta şasi, motor ve tekerlekleri üretip birleştir. Yolda daha hızlı gider.
- **Kayıt:** Oyun her 3 dakikada ve uyuduğunda otomatik kaydedilir.

## Ayarlar
Ana menüde veya duraklatma menüsünde: grafik kalitesi ön ayarları, gölge çözünürlüğü, çim yoğunluğu, görüş mesafesi,
ortam gölgelemesi (AO), bloom, FXAA, tam ekran, FPS göstergesi, fare hassasiyeti, FOV, kafa sallanması, ses düzeyi.
Bilgisayarın zorlanırsa **Grafik Kalitesi: Düşük** seç.

## Grafik teknikleri
Doğrusal renk uzayında aydınlatma ve ACES ton eşleme, PCF yumuşak gölgeler, yüzey gradyanı ile kabartma (bump),
yaprakların ışığı geçirmesi (yarı saydamlık), kapsama korumalı mipmap'li alfa kesimli yapraklar ve çim, yükseklik tabanlı
arazi doku karışımı, triplanar kaya, atmosferik sis ve güneş parıltısı, prosedürel bulutlu gökyüzü, derinliğe göre renk
değiştiren göl suyu, gün-gece döngüsü, yağmur ve ıslaklık, noktasal ışıklar (fener, ateş, far), ekran sonrası FXAA,
bloom, SSAO ve vinyet.
