from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import numpy as np
import os
import shutil
import sys
import cv2  # Añadimos opencv para el nuevo método

# Definir la ruta de la fuente usando os.path para mayor compatibilidad
FUENTE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font", "DIN1451-36breit.ttf")

# Definir una fuente adicional para el dígito "0" - usaremos una sin diagonal
# Varias opciones: Arial, Verdana o cualquier otra fuente sans-serif
FUENTE_CERO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font", "arial.ttf")

# Definir una fuente para la letra "I" que tenga serifs o trazos horizontales
FUENTE_I_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font", "times.ttf")

# Constantes para el tamaño de entrada de las imágenes sintéticas
INPUT_WIDTH = 200
INPUT_HEIGHT = 50

def generar_caracter(caracter="A", output_dir="dades"):
    """Genera una imagen base con un solo carácter."""
    # Crear imagen más grande para luego recortarla exactamente al carácter
    img = Image.new('L', (64, 64), color=255)
    d = ImageDraw.Draw(img)
    
    # Seleccionar fuente según el carácter
    if caracter == "0":
        try:
            # Intentar usar una fuente que tenga un 0 más rectangular
            font = ImageFont.truetype(FUENTE_CERO_PATH, 40)
        except IOError:
            # Si no se puede cargar la fuente especial, usar la fuente normal
            try:
                font = ImageFont.truetype(FUENTE_PATH, 40)
            except IOError:
                font = ImageFont.load_default()
    elif caracter == "I":
        try:
            # Usar una fuente con serifs para la letra I
            font = ImageFont.truetype(FUENTE_I_PATH, 40)
        except IOError:
            # Si no se puede cargar la fuente especial, crear una I modificada
            try:
                font = ImageFont.truetype(FUENTE_PATH, 40)
            except IOError:
                font = ImageFont.load_default()
    else:
        # Para cualquier otro carácter, usar la fuente normal
        try:
            font = ImageFont.truetype(FUENTE_PATH, 40)
        except IOError:
            print(f"⚠️ No se pudo cargar la fuente {FUENTE_PATH}, usando fuente por defecto")
            font = ImageFont.load_default()
    
    # Obtenemos dimensiones del texto
    try:
        # Para versiones más nuevas de Pillow
        bbox = font.getbbox(caracter)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Para versiones anteriores de Pillow
        text_width, text_height = font.getsize(caracter)
    
    # Centrar el carácter en la imagen
    position = ((64-text_width)//2, (64-text_height)//2)
    
    # Dibujar el carácter
    d.text(position, caracter, fill=0, font=font)
    
    # Si es una I y la fuente especial no está disponible, modificar manualmente
    if caracter == "I" and font != ImageFont.truetype(FUENTE_I_PATH, 40):
        # Encontrar el bounding box de la I
        img_array = np.array(img)
        rows = np.any(img_array < 255, axis=1)
        cols = np.any(img_array < 255, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Añadir trazos horizontales arriba y abajo para simular una I con serifs
            line_width = int((x_max - x_min) * 2.5)  # Línea horizontal más ancha que la vertical
            center_x = (x_min + x_max) // 2
            start_x = max(0, center_x - line_width // 2)
            end_x = min(img.width - 1, center_x + line_width // 2)
            
            # Trazar línea superior
            for x in range(start_x, end_x + 1):
                for y in range(y_min, y_min + 3):  # Grosor de 3 píxeles
                    if 0 <= y < img.height and 0 <= x < img.width:
                        img_array[y, x] = 0
            
            # Trazar línea inferior
            for x in range(start_x, end_x + 1):
                for y in range(y_max - 2, y_max + 1):  # Grosor de 3 píxeles
                    if 0 <= y < img.height and 0 <= x < img.width:
                        img_array[y, x] = 0
            
            # Actualizar la imagen
            img = Image.fromarray(img_array)
    
    # Recortar la imagen al área exacta del carácter (con un pequeño margen)
    if caracter != " ":  # Evitar problemas con espacios
        # Encontrar el bounding box real del carácter en la imagen
        img_array = np.array(img)
        rows = np.any(img_array < 255, axis=1)
        cols = np.any(img_array < 255, axis=0)
        
        if np.any(rows) and np.any(cols):  # Asegurarse de que hay contenido visible
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Añadir un pequeño margen (2 píxeles)
            margin = 6
            y_min = max(0, y_min - margin)
            y_max = min(img.height - 1, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(img.width - 1, x_max + margin)
            
            # Recortar la imagen
            img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Redimensionar a 32x32 para tamaño final estándar
    img = img.resize((32, 32), Image.BICUBIC)
    
    return img

def aplicar_augmentaciones(img):
    """Aplica varias transformaciones y filtros a la imagen."""
    # Generamos una imagen de tamaño aleatorio para luego redimensionarla
    # Reducimos el rango de tamaños a valores más conservadores
    random_size = random.choice([28, 32, 36])  # Reducido desde [24, 28, 32, 36, 40, 48]
    if img.size != (random_size, random_size):
        img = img.resize((random_size, random_size), Image.BICUBIC)
    
    width, height = img.size
    # Reducimos el padding para permitir rotaciones menos agresivas
    padding = int(max(width, height) * 0.2)  # Reducido de 0.3 (30%) a 0.2 (20%)
    padded_img = Image.new('L', (width + 2*padding, height + 2*padding), color=255)
    padded_img.paste(img, (padding, padding))
    img = padded_img

    # Rotación aleatoria con ángulo menos agresivo
    angle = random.uniform(-15, 15)  # Reducido de [-30, 30] a [-15, 15]
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Recortar al tamaño original después de la rotación
    center_x, center_y = img.size[0] // 2, img.size[1] // 2
    img = img.crop((
        center_x - width // 2,
        center_y - height // 2,
        center_x + width // 2,
        center_y + height // 2
    ))

    # Transformación de perspectiva menos agresiva
    if random.choice([True, False]):
        width, height = img.size
        
        # Define distorsión máxima (reducida al 10% de la dimensión)
        max_shift = min(width, height) * 0.1  # Reducido de 0.2 a 0.1
        
        # Esquinas originales
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        
        # Nuevas esquinas con desplazamientos aleatorios menos pronunciados
        new_corners = [
            (random.uniform(0, max_shift), random.uniform(0, max_shift)),
            (width - random.uniform(0, max_shift), random.uniform(0, max_shift)),
            (width - random.uniform(0, max_shift), height - random.uniform(0, max_shift)),
            (random.uniform(0, max_shift), height - random.uniform(0, max_shift))
        ]
        
        # Calcular coeficientes de transformación de perspectiva
        coeffs = np.linalg.solve(
            np.array([[x, y, 1, 0, 0, 0, -x*x2, -y*x2] for (x, y), (x2, y2) in zip(corners, new_corners)] +
                   [[0, 0, 0, x, y, 1, -x*y2, -y*y2] for (x, y), (x2, y2) in zip(corners, new_corners)]),
            np.array([x2 for (x2, y2) in new_corners] + [y2 for (x2, y2) in new_corners])
        )
        
        # Aplicar transformación
        img = img.transform(
            size=(width, height),
            method=Image.PERSPECTIVE,
            data=tuple(coeffs),
            resample=Image.BICUBIC,
            fill=1
        )

    # Desenfoque gaussiano menos pronunciado
    if random.choice([True, False]):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))  # Reducido de [0.5, 1.0] a [0.3, 0.7]
    
    # Ajuste de contraste y brillo menos extremo
    if random.choice([True, False]):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.2))  # Cambiado de [0.5, 1.0] a [0.7, 1.2]
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.2))  # Cambiado de [0.5, 1.0] a [0.7, 1.2]
    
    # Añadir ruido gaussiano menos pronunciado
    if random.choice([True, False]):
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 8, np_img.shape)  # Reducido de 15 a 8
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
    
    # Añadir posibilidad de escala aleatoria menos agresiva
    if random.choice([True, False]):
        scale_x = random.uniform(0.9, 1.1)  # Reducido de [0.8, 1.2] a [0.9, 1.1]
        scale_y = random.uniform(0.9, 1.1)  # Reducido de [0.8, 1.2] a [0.9, 1.1]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        img = img.resize((new_width, new_height), Image.BICUBIC)
        # Volver al tamaño original después de estirar/comprimir
        img = img.resize((width, height), Image.BICUBIC)
    
    # Al final de todas las transformaciones, asegurar que el carácter ocupa toda la imagen
    # Encontrar el bounding box real del contenido
    img_array = np.array(img)
    rows = np.any(img_array < 255, axis=1)
    cols = np.any(img_array < 255, axis=0)
    
    # Solo ajustar si hay contenido visible
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Recortar al área exacta del contenido
        if x_min < x_max and y_min < y_max:
            img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            
            # Redimensionar a 32x32
            img = img.resize((32, 32), Image.BICUBIC)
    else:
        # Si no hay contenido visible, simplemente redimensionar
        img = img.resize((32, 32), Image.BICUBIC)
    
    return img

def guardar_imagen(img, caracter, idx, label=None):
    """Guarda la imagen con un nombre único según el carácter y el índice."""
    # Define el directorio de salida a partir de este script
    output_dir = os.path.join(os.path.dirname(__file__), "dades")
    
    # Si se especifica una etiqueta, usar un subdirectorio por clase
    if label is not None:
        output_dir = os.path.join(output_dir, label)
    
    # Si el directorio no existe, lo crea
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Asegurar que la imagen tiene el tamaño correcto antes de guardar
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BICUBIC)
    
    # Guarda la imagen
    img.save(f"{output_dir}/caracter_{caracter}_{idx}.jpg")

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

def generar_datos(num_augmentations=10, organizar_por_clase=True):
    """Genera imágenes de todos los caracteres (letras y números)."""
    # Crea la carpeta si no existe
    output_dir = os.path.join(os.path.dirname(__file__), "dades")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Caracteres a generar (números y letras mayúsculas)
    caracteres = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    print(f"🔄 Generando imágenes para {len(caracteres)} caracteres con {num_augmentations} augmentaciones cada uno...")
    
    total_generadas = 0
    
    for caracter in caracteres:
        # Para cada carácter, generamos una imagen base y varias augmentaciones
        base_img = generar_caracter(caracter)
        
        # Guardar imagen base
        label = caracter if organizar_por_clase else None
        guardar_imagen(base_img, caracter, 0, label)
        total_generadas += 1
        
        # Generar y guardar augmentaciones
        for j in range(1, num_augmentations):
            aug_img = aplicar_augmentaciones(base_img)
            guardar_imagen(aug_img, caracter, j, label)
            total_generadas += 1
    
    print(f"✅ Se han generado {total_generadas} imágenes correctamente.")

def create_synthetic_dataset(base_dir, num_samples=5000):
    """Crea conjunts de dades sintètics a partir de caràcters individuales"""
    X = []  # imatges
    y = []  # etiquetes (text)
    
    # Llista de caràcters disponibles (directoris al base_dir)
    available_chars = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not available_chars:
        print(f"❌ No se encontraron caracteres en {base_dir}. Asegúrate de generar primero las imágenes individuales.")
        return None, None
    
    print(f"🔄 Generando {num_samples} secuencias sintéticas usando caracteres de {len(available_chars)} clases...")
    
    for i in range(num_samples):
        if i % 500 == 0 and i > 0:
            print(f"  ↪ Generadas {i} secuencias...")
            
        # Determina aleatòriament la longitud de la seqüència (3-9 caràcters)
        seq_len = random.randint(3, 9)
        
        # Selecciona caràcters aleatoris
        seq_chars = random.choices(available_chars, k=seq_len)
        
        # Construeix la imatge composta
        images = []
        valid_chars = []  # Solo guardar caracteres con imágenes válidas
        
        for char in seq_chars:
            # Selecciona una imatge aleatòria del caràcter
            char_dir = os.path.join(base_dir, char)
            img_files = [f for f in os.listdir(char_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files:
                continue
            
            try:
                img_path = os.path.join(char_dir, random.choice(img_files))
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Cargar directamente en escala de grises
                
                if img is None:
                    continue  # Skip if image couldn't be loaded
                
                # Asegurar que solo tenemos el carácter, sin espacios blancos
                if img.shape[0] != 32 or img.shape[1] != 32:
                    # Encontrar el bounding box del carácter
                    rows = np.any(img < 255, axis=1)
                    cols = np.any(img < 255, axis=0)
                    
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        # Recortar al contenido
                        img = img[y_min:y_max+1, x_min:x_max+1]
                
                # Redimensionar a tamaño consistente para secuencia
                img = cv2.resize(img, (25, 50))
                
                # Afegeix variacions aleatòries (menos agresivas)
                if random.random() > 0.5:
                    angle = random.uniform(-3, 3)  # Reducido de [-5, 5] a [-3, 3]
                    M = cv2.getRotationMatrix2D((12, 25), angle, 1)
                    img = cv2.warpAffine(img, M, (25, 50))
                
                images.append(img)
                valid_chars.append(char)
            except Exception as e:
                print(f"Error procesando carácter {char}: {e}")
                continue
        
        # Verificar que tenemos al menos un carácter
        if not images or not valid_chars:
            continue
            
        # Crear imagen con el espacio exacto que necesitamos
        total_width = sum(img.shape[1] for img in images) + (len(images) - 1) * 4  # 4px de espacio entre caracteres
        seq_img = np.ones((50, total_width), dtype=np.uint8) * 255
        
        # Colocar cada imagen con un espacio fijo entre ellas
        x_offset = 0
        for img in images:
            h, w = img.shape
            # Asegurarnos de que hay suficiente espacio
            if x_offset + w <= total_width:
                seq_img[:h, x_offset:x_offset+w] = img
                x_offset += w + 4  # 4px de espacio fijo entre caracteres
        
        # Aplica transformacions globals (menos agresivas)
        # Rotació lleugera
        angle = random.uniform(-5, 5)  # Reducido de [-10, 10] a [-5, 5]
        M = cv2.getRotationMatrix2D((total_width//2, 25), angle, 1)
        seq_img = cv2.warpAffine(seq_img, M, (total_width, 50), borderValue=255)
        
        # Perspectiva aleatòria (menos agresiva y menos frecuente)
        if random.random() > 0.8:  # Reducido de 0.7 a 0.8 (menos frecuente)
            pts1 = np.float32([[0,0], [total_width,0], [0,50], [total_width,50]])
            pts2 = np.float32([
                [random.randint(0,5), random.randint(0,3)],  # Reducidos de 10/5 a 5/3
                [total_width-random.randint(0,5), random.randint(0,3)],
                [random.randint(0,5), 50-random.randint(0,3)],
                [total_width-random.randint(0,5), 50-random.randint(0,3)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            seq_img = cv2.warpPerspective(seq_img, M, (total_width, 50), borderValue=255)
        
        # Redimensiona a l'amplada fixa d'entrada
        seq_img = cv2.resize(seq_img, (INPUT_WIDTH, INPUT_HEIGHT))
        
        # Normalitza i afegeix canal
        seq_img = seq_img / 255.0
        seq_img = np.expand_dims(seq_img, axis=-1)
        
        # Afegeix soroll (menos intenso)
        if random.random() > 0.8:
            noise = np.random.randn(INPUT_HEIGHT, INPUT_WIDTH, 1) * 0.05  # Reducido de 0.1 a 0.05
            seq_img = np.clip(seq_img + noise, 0, 1)
        
        # Afegeix la imatge i l'etiqueta als conjunts de dades
        X.append(seq_img)
        y.append(''.join(valid_chars))
    
    print(f"✅ Se han generado {len(X)} secuencias sintéticas correctamente.")
    return np.array(X), y

def crear_dataset_sintetico():
    """Función auxiliar para crear y guardar un dataset sintético"""
    # Especifica el directorio de imágenes individuales
    base_dir = os.path.join(os.path.dirname(__file__), "dades")
    
    if not os.path.exists(base_dir):
        print(f"❌ El directorio {base_dir} no existe. Genera primero las imágenes individuales.")
        return
    
    try:
        # Mejor manejo de entrada con valores por defecto
        num_input = input("Número de secuencias sintéticas a generar [1000]: ")
        num_samples = 1000  # valor por defecto
        
        if num_input.strip():  # Si hay algún valor
            try:
                num_samples = int(num_input)
            except ValueError:
                print("⚠️ Valor no válido, usando el valor por defecto (1000)")
        
        # Genera el dataset
        X, y = create_synthetic_dataset(base_dir, num_samples)
        
        if X is None:
            return
        
        # Crea directorio para guardar el dataset
        output_dir = os.path.join(os.path.dirname(__file__), "dataset_sintetico")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Guarda algunas imágenes de ejemplo
        print("📁 Guardando algunas imágenes de ejemplo...")
        for i in range(min(20, len(X))):
            img = (X[i] * 255).astype(np.uint8).squeeze()
            cv2.imwrite(os.path.join(output_dir, f"ejemplo_{i}_{y[i]}.jpg"), img)
        
        # Guarda el dataset completo en formato numpy
        print("💾 Guardando dataset completo...")
        np.save(os.path.join(output_dir, "X_data.npy"), X)
        np.save(os.path.join(output_dir, "y_labels.npy"), np.array(y, dtype=object))
        
        print(f"✅ Dataset guardado en {output_dir}")
        
    except Exception as e:
        print(f"❌ Error al crear el dataset sintético: {str(e)}")
        import traceback
        traceback.print_exc()

def mostrar_menu():
    """Muestra un menú interactivo en la terminal."""
    while True:
        print("\n=== 🔠 Generador de Imágenes de Caracteres ===")
        print("1. Generar datos de caracteres individuales")
        print("2. Crear dataset sintético de secuencias")
        print("3. Eliminar carpeta de datos")
        print("4. Salir")
        
        opcion = input("\nSeleccione una opción (1-4): ")
        
        if opcion == "1":
            try:
                num_augmentations = int(input("Número de augmentaciones por carácter [10]: ") or "10")
                organizar = input("¿Organizar por clase en subcarpetas? (s/n) [s]: ").lower() != "n"
                generar_datos(num_augmentations, organizar)
            except ValueError:
                print("❌ Por favor, introduce números válidos.")
        elif opcion == "2":
            crear_dataset_sintetico()
        elif opcion == "3":
            confirmar = input("¿Estás seguro de que deseas eliminar todos los datos? (s/n): ")
            if confirmar.lower() == "s":
                eliminar_carpeta_dades()
        elif opcion == "4":
            print("👋 ¡Hasta pronto!")
            break
        else:
            print("❌ Opción no válida. Por favor, elija una opción entre 1 y 4.")

if __name__ == "__main__":
    mostrar_menu()
