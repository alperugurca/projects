import random

def tas_kagit_makas_alper_ugurcan():
    """
    Taş, Kağıt, Makas oyununu yöneten ana fonksiyon.
    """
    secenekler = ["taş", "kağıt", "makas"]
    oyun_sayisi = 0
    
    print("Taş, Kağıt, Makas oyununa hoş geldiniz!")
    print("Kurallar:")
    print("- Her oyun en fazla üç turdan oluşur.")
    print("- İlk iki turu kazanan oyunun galibi olur.")
    print("- Seçiminizi yapmak için 'taş', 'kağıt' veya 'makas' yazın.")
    print("- Oyundan çıkmak için 'q' tuşuna basın.")

    while True:
        oyuncu_galibiyet = 0
        bilgisayar_galibiyet = 0
        tur_sayisi = 0

        while oyuncu_galibiyet < 2 and bilgisayar_galibiyet < 2:
            tur_sayisi += 1
            print(f"\n--- Tur {tur_sayisi} ---")

            # Oyuncu seçimi
            while True:
                oyuncu_secimi = input("Seçiminizi yapın (taş/kağıt/makas): ").lower()
                if oyuncu_secimi == 'q':
                    print("Oyundan çıkılıyor...")
                    return
                if oyuncu_secimi in secenekler:
                    break
                print("Geçersiz seçim. Lütfen tekrar deneyin.")

            # Bilgisayar seçimi
            bilgisayar_secimi = random.choice(secenekler)

            print(f"Sizin seçiminiz: {oyuncu_secimi}")
            print(f"Bilgisayarın seçimi: {bilgisayar_secimi}")

            # Kazananı belirleme
            if oyuncu_secimi == bilgisayar_secimi:
                print("Berabere!")
            elif (
                (oyuncu_secimi == "taş" and bilgisayar_secimi == "makas") or
                (oyuncu_secimi == "kağıt" and bilgisayar_secimi == "taş") or
                (oyuncu_secimi == "makas" and bilgisayar_secimi == "kağıt")
            ):
                print("Bu turu siz kazandınız!")
                oyuncu_galibiyet += 1
            else:
                print("Bu turu bilgisayar kazandı!")
                bilgisayar_galibiyet += 1

            print(f"Skor: Siz {oyuncu_galibiyet} - {bilgisayar_galibiyet} Bilgisayar")

        # Oyun sonucu
        oyun_sayisi += 1
        if oyuncu_galibiyet > bilgisayar_galibiyet:
            print("\nTebrikler! Oyunu kazandınız!")
        else:
            print("\nÜzgünüm, oyunu bilgisayar kazandı.")

        # Devam etme isteği
        devam_oyuncu = input("Başka bir oyun oynamak ister misiniz? (e/h): ").lower() == 'e'
        devam_bilgisayar = random.choice([True, False])

        if devam_bilgisayar:
            print("Bilgisayar: Evet, tekrar oynamak istiyorum!")
        else:
            print("Bilgisayar: Hayır, teşekkürler. Başka zaman oynayalım.")

        if devam_oyuncu and devam_bilgisayar:
            print("Harika! Yeni bir oyun başlıyor...")
        else:
            print(f"Oyun bitti. Toplam {oyun_sayisi} oyun oynadınız.")
            break

if __name__ == "__main__":
    tas_kagit_makas_alper_ugurcan()
