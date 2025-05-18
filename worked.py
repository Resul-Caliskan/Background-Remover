# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:36:02 2025
@author: resul
"""
import cv2
import matplotlib.pyplot as plt
from dis_bg_remover import remove_background
from PIL import Image
from pathlib import Path
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
import numpy as np
import cv2

def detect_object_bounds(image, debug=True):
    """
    Şeffaf arka planlı bir görseldeki nesnenin sınırlarını tespit eder.
    
    Parametre:
    - image: PIL.Image nesnesi (RGBA modunda)
    - debug: True ise görsel ve alpha maskesi gösterilir
    
    Dönüş Değeri:
    - (left, top, right, bottom): Nesnenin sınır koordinatları
    """
    # Görüntüyü NumPy dizisine dönüştür
    img_array = np.array(image)
    
    # Alfa kanalını kullanarak maskeleme
    alpha_channel = img_array[:, :, 3]
    mask = alpha_channel > 128
    
    if not np.any(mask):
        return None
    
    # Sınırları bul
    rows_with_object = np.any(mask, axis=1)
    cols_with_object = np.any(mask, axis=0)
    
    top = np.argmax(rows_with_object)
    bottom = len(rows_with_object) - np.argmax(rows_with_object[::-1]) - 1
    left = np.argmax(cols_with_object)
    right = len(cols_with_object) - np.argmax(cols_with_object[::-1]) - 1
    
    if debug:
        import matplotlib.pyplot as plt
        print("→ detect_object_bounds() içinde kullanılan görüntü ve alpha maskesi:")
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("RGBA Görüntü")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Alpha Maskesi")
        plt.axis('off')
        
        plt.show()
    
    return (left, top, right, bottom)





def apply_daylight_effect(image, brightness_factor=1.1, contrast_factor=1.05, saturation_factor=1.1, warmth_shift=15):
    """
    Photoshop benzeri, daha doğal gün ışığı efekti:
    - Parlaklık ve kontrast artırımı
    - Hafif sıcaklık (white balance)
    - Saturation artışı
    - Metalik alanların korunması
    """
    img_array = np.array(image).astype(float)

    has_alpha = img_array.shape[2] == 4
    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        img_array = img_array[:, :, :3]

    # Gümüş alanları tespit etmek için bir maske: düşük saturasyon + yüksek parlaklık
    hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    silver_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 150)

    # 1. Parlaklık & Kontrast
    img_array = np.clip(img_array * brightness_factor, 0, 255)
    mean_intensity = np.mean(img_array)
    img_array = (img_array - mean_intensity) * contrast_factor + mean_intensity
    img_array = np.clip(img_array, 0, 255)

    # 2. White balance - sıcak ton (R ve G kanallarına ekleme)
    warmth_mask = np.ones_like(img_array[:, :, 0], dtype=float)
    warmth_mask[silver_mask] = 0.2  # gümüş alanlara daha az ısı uygula

    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + warmth_shift * 0.4 * warmth_mask, 0, 255)  # R
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + warmth_shift * 0.2 * warmth_mask, 0, 255)  # G

    # 3. Saturation
    hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(float)

    # 4. Highlight boost (gümüş alanlarda uygulama)
    highlight_mask = (img_array > 180)
    img_array = np.where(highlight_mask, np.clip(img_array * 1.05, 0, 255), img_array)

    # # 5. Vignette (opsiyonel, yumuşak)
    # rows, cols = img_array.shape[:2]
    # kernel_x = cv2.getGaussianKernel(cols, cols / 2.5)
    # kernel_y = cv2.getGaussianKernel(rows, rows / 2.5)
    # kernel = kernel_y * kernel_x.T
    # mask = kernel / kernel.max()
    # vignette_strength = 0.4
    # for i in range(3):
    #     img_array[:, :, i] = img_array[:, :, i] * (1 - vignette_strength + vignette_strength * mask)

    img_array = np.clip(img_array, 0, 255)

    if has_alpha:
        img_array = np.dstack((img_array, alpha_channel))

    return Image.fromarray(img_array.astype(np.uint8))



def center_and_resize_object(input_path, output_path, target_size=(1600, 1600), 
                            size_factor=1.25, brightness_factor=1.2, background_color=(255, 255, 255, 255)):
    """
    Güncellenmiş fonksiyon: Photoshop benzeri gün ışığı efekti eklendi
    """
    try:
        start_time = time.time()
        
        # Görüntüyü aç
        if isinstance(input_path, str):  # Dosya yolu verilmişse
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Dosya bulunamadı: {input_path}")
            input_img = Image.open(input_path)
        else:  # PIL Image nesnesi verilmişse
            input_img = input_path
        
        # RGBA moduna dönüştür (şeffaflık için)
        if input_img.mode != 'RGBA':
            input_img = input_img.convert('RGBA')
        
        # Nesne sınırlarını tespit et
        bounds = detect_object_bounds(input_img)
        if not bounds:
            raise ValueError("Görselde nesne tespit edilemedi veya alpha kanalı yok")
        
        left, top, right, bottom = bounds
        
        # Nesnenin boyutlarını hesapla
        object_width = right - left
        object_height = bottom - top
        
        # Nesnenin merkezini bul
        object_center_x = (left + right) // 2
        object_center_y = (top + bottom) // 2
        
        print(f"Nesne sınırları: Sol={left}, Üst={top}, Sağ={right}, Alt={bottom}")
        print(f"Nesne boyutları: {object_width}x{object_height} piksel")
        print(f"Nesne merkezi: ({object_center_x}, {object_center_y})")
        
        # Yeni boş görüntü oluştur (hedef boyutta)
        new_width, new_height = target_size
        new_img = Image.new('RGBA', (new_width, new_height), background_color)
        
        # Nesnenin en büyük boyutunu bul
        max_object_dimension = max(object_width, object_height)
        
        # Nesneyi hedef boyuta göre ölçekle ve istenen büyüklük faktörünü uygula
        scale_factor = min(new_width, new_height) / max_object_dimension * size_factor
        
        print(f"Büyüklük faktörü: {size_factor} (fiili ölçek: {scale_factor:.2f})")
        
        # Nesneyi yeniden boyutlandır
        new_object_width = int(object_width * scale_factor)
        new_object_height = int(object_height * scale_factor)
        
        print(f"Yeni nesne boyutları: {new_object_width}x{new_object_height} piksel")
        
        # Orijinal nesneden kırp
        cropped_img = input_img.crop((left, top, right, bottom))
        
        # Kırpılmış nesneyi yeniden boyutlandır
        resized_img = cropped_img.resize((new_object_width, new_object_height), Image.LANCZOS)
        
        # Gün ışığı efekti uygula (Photoshop benzeri)
        if brightness_factor != 1.0:
            resized_img = apply_daylight_effect(
                resized_img,
                brightness_factor=brightness_factor,
                contrast_factor=1.05,
                saturation_factor=1.15,
                warmth_shift=15  # Hafif sıcaklık
            )
            print(f"Gün ışığı efekti uygulandı, parlaklık faktörü: {brightness_factor:.2f}")
        
        # Hedef görüntünün merkezini hesapla
        target_center_x = new_width // 2
        target_center_y = new_height // 2
        
        # Nesneyi hedef görüntünün ortasına yerleştir
        paste_x = target_center_x - (new_object_width // 2)
        paste_y = target_center_y - (new_object_height // 2)
        
        # Nesneyi yapıştır
        new_img.paste(resized_img, (paste_x, paste_y), resized_img)
        
        # Çıktı klasörünü kontrol et ve gerekirse oluştur
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Sonucu kaydet
        new_img.save(output_path)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Nesne merkezlendi ve yeniden boyutlandırıldı: {output_path} ({elapsed_time:.2f} saniye)")
        
        return True, output_path
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False, str(e)
# Ana kod buradan başlıyor
# Dönüştürülecek görüntü yolu
input_path = "valiz.DNG"
output_path = "valiz.png"

# Görüntüyü aç ve dönüştür
try:
    input_img = Image.open(input_path)
    # Görüntü formatını kontrol et ve dönüştür
    if input_img.mode != 'RGB' and input_img.mode != 'RGBA':
        input_img = input_img.convert('RGB')
        print(f"Görüntü {input_img.mode} formatından RGB'ye dönüştürüldü")
    
    # Dosya tipini kontrol et ve gerekirse dönüştür
    file_ext = Path(input_path).suffix.lower()
    if file_ext in ['.dng', '.cr2', '.nef', '.arw']:
        print(f"RAW format algılandı: {file_ext}, dönüştürülüyor...")
        input_img = input_img.convert('RGB')
    
    # PNG olarak kaydet
    input_img.save(output_path)
    print(f"Dönüştürülen görüntü {output_path} olarak kaydedildi.")

except Exception as e:
    print(f"Görüntü açılırken bir hata oluştu: {e}")

# Arkaplan kaldırma işlemi
model_path = ".\\models\\isnet_dis.onnx"
image_path = output_path  # Artık PNG dosyası

# Orijinal görüntüyü göster
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title("Orijinal Görüntü")
plt.axis('off')
plt.show()

# Arkaplanı kaldır
img, mask = remove_background(model_path, image_path)

# Arka plan kaldırılmış görüntüyü kaydet
transparent_output = 'Bavull.png'
cv2.imwrite(transparent_output, img)
cv2.imwrite('mask.jpg', mask)

# Maske ve sonuçları göster
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
ax1.set_title("Girdi Görseli")
ax1.axis('off')

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
ax2.set_title("Maske")
ax2.axis('off')
plt.show()

# Arka planı kaldırılmış PNG görselini PIL ile aç
transparent_img = Image.open(transparent_output)
transparent_img.save("transparent_saved.png")
# Görsel RGBA modunda değilse dönüştür
if transparent_img.mode != 'RGBA':
    transparent_img = transparent_img.convert('RGBA')

# Merkezleme, boyutlandırma ve parlaklık ayarlaması yapılacak sonuç dosyası
centered_output = 'bavul3.png'

# Nesneyi merkezle ve yeniden boyutlandır
success, output_file = center_and_resize_object(
    transparent_img,  # Doğrudan PIL Image nesnesi kullanıyoruz
    centered_output,
    target_size=(1600, 1600),
    size_factor=0.94,  # Nesneyi %25 daha büyük göster
    brightness_factor=1.05,  # Parlaklığı %20 artır
    background_color=(255, 255, 255, 255)  # Beyaz arka plan (RGBA)
)

if success:
    # Son görüntüyü göster
    final_img = Image.open(centered_output)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(final_img))
    plt.title("Son İşlenmiş Görüntü (Merkezlenmiş ve Boyutlandırılmış)")
    plt.axis('off')
    plt.show()
else:
    print(f"Merkezleme işlemi başarısız oldu: {output_file}")

print("İşlem tamamlandı.")