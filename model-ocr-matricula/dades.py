from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import numpy as np
import os
import shutil
import sys

# Definir la ruta de la fuente usando os.path para mayor compatibilidad
FUENTE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font", "DIN1451-36breit.ttf")

def generar_matricula(texto="1234 ABC", output_dir="PIA/Proyecto2-Parquing/model-ocr/dades"):
    # Genera la imagen base de la matrícula
    # Reducimos la altura a 50px y ajustamos el ancho proporcionalmente
    img = Image.new('RGB', (140, 50), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FUENTE_PATH, 50)  # Usamos la fuente DIN1451 similar a las matrículas reales
    except IOError:
        print(f"⚠️ No se pudo cargar la fuente {FUENTE_PATH}, usando fuente por defecto")
        font = ImageFont.load_default()
    d.text((10, 10), texto, fill=(0, 0, 0), font=font)
    return img

def aplicar_augmentaciones(img):
    # Añadir padding antes de rotaciones para evitar recortes
    width, height = img.size
    padding = int(max(width, height) * 0.2)  # 20% de padding
    padded_img = Image.new('RGB', (width + 2*padding, height + 2*padding), color=(255, 255, 255))
    padded_img.paste(img, (padding, padding))
    img = padded_img

    # Rotación aleatoria con ángulo más sutil
    angle = random.uniform(-3, 3)  # Reducimos aún más el ángulo para evitar rotaciones excesivas
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Recortar al tamaño original después de la rotación
    center_x, center_y = img.size[0] // 2, img.size[1] // 2
    img = img.crop((
        center_x - width // 2,
        center_y - height // 2,
        center_x + width // 2,
        center_y + height // 2
    ))

    # Transformación de perspectiva
    if random.choice([True, False]):
        width, height = img.size
        
        # Define corners with subtle distortion (max 20% of dimension)
        max_shift = min(width, height) * 0.2
        
        # Original corners
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        
        # New corners with subtle random shifts
        new_corners = [
            (random.uniform(0, max_shift), random.uniform(0, max_shift)),
            (width - random.uniform(0, max_shift), random.uniform(0, max_shift)),
            (width - random.uniform(0, max_shift), height - random.uniform(0, max_shift)),
            (random.uniform(0, max_shift), height - random.uniform(0, max_shift))
        ]
        
        # Calculate perspective transform coefficients
        coeffs = np.linalg.solve(
            np.array([[x, y, 1, 0, 0, 0, -x*x2, -y*x2] for (x, y), (x2, y2) in zip(corners, new_corners)] +
                   [[0, 0, 0, x, y, 1, -x*y2, -y*y2] for (x, y), (x2, y2) in zip(corners, new_corners)]),
            np.array([x2 for (x2, y2) in new_corners] + [y2 for (x2, y2) in new_corners])
        )
        
        # Apply transform
        img = img.transform(
            size=(width, height),
            method=Image.PERSPECTIVE,
            data=tuple(coeffs),
            resample=Image.BICUBIC,
            fill=1
        )

    # Desenfoque gaussiano
    if random.choice([True, False]):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    # Ajuste de contraste y brillo
    if random.choice([True, False]):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Añade ruido gaussiano
    if random.choice([True, False]):
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
    return img

def guardar_imagen(img, texto, idx):
    # Define el directorio de salida a partir de este script
    output_dir = os.path.join(os.path.dirname(__file__), "dades")
    # Si el directorio no existe, lo crea
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Guarda la imagen con un nombre único
    img.save(f"{output_dir}/matricula_{texto}_{idx}.jpg")

def eliminar_carpeta_dades():
    """Elimina la carpeta de datos si existe."""
    output_dir = os.path.join(os.path.dirname(__file__), "dades")
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"✅ Carpeta '{output_dir}' eliminada con éxito.")
        except Exception as e:
            print(f"❌ Error al eliminar la carpeta: {e}")
    else:
        print(f"⚠️ La carpeta '{output_dir}' no existe.")

def generar_datos(num_samples=10, augmentations_per_sample=5):
    """Genera los datos de las matrículas."""
    # Crea la carpeta si no existe
    output_dir = os.path.join(os.path.dirname(__file__), "dades")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    allowed_letters = "BCDFGHJKLMNPQRSTVWXYZ"  # Sin vocales
    
    print(f"🔄 Generando {num_samples} matrículas con {augmentations_per_sample-1} augmentaciones cada una...")
    
    for i in range(num_samples):
        # Genera texto de matrícula: 4 dígitos y 3 letras (sin vocales)
        texto = f"{random.randint(1000,9999)} " + "".join([random.choice(allowed_letters) for _ in range(3)])
        base_img = generar_matricula(texto)
        guardar_imagen(base_img, texto, 0)
        for j in range(1, augmentations_per_sample):
            aug_img = aplicar_augmentaciones(base_img)
            guardar_imagen(aug_img, texto, j)
    
    total_imgs = num_samples if augmentations_per_sample == 0 else num_samples * augmentations_per_sample
    print(f"✅ Se han generado {total_imgs} imágenes correctamente.")

def mostrar_menu():
    """Muestra un menú interactivo en la terminal."""
    while True:
        print("\n=== 🚗 Generador de Matrículas ===")
        print("1. Generar datos")
        print("2. Eliminar carpeta de datos")
        print("3. Salir")
        
        opcion = input("\nSeleccione una opción (1-3): ")
        
        if opcion == "1":
            try:
                num_samples = int(input("Número de matrículas base a generar [10]: ") or "10")
                augmentations = int(input("Número de augmentaciones por matrícula [5]: ") or "5")
                generar_datos(num_samples, augmentations)
            except ValueError:
                print("❌ Por favor, introduce números válidos.")
        elif opcion == "2":
            confirmar = input("¿Estás seguro de que deseas eliminar todos los datos? (s/n): ")
            if confirmar.lower() == "s":
                eliminar_carpeta_dades()
        elif opcion == "3":
            print("👋 ¡Hasta pronto!")
            break
        else:
            print("❌ Opción no válida. Por favor, elija una opción entre 1 y 3.")

if __name__ == "__main__":
    mostrar_menu()
