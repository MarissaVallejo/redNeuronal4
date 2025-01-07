import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def split_image_blocks(image, num_blocks=9):
    """Divide una imagen en bloques.

    Args:
        image: PIL Image
        num_blocks: Número total de bloques (debe ser un cuadrado perfecto)

    Returns:
        blocks: Lista de bloques de imagen
        positions: Lista de tuplas (fila, columna)
        grid_size: Tupla (filas, columnas)
    """
    width, height = image.size

    # Calcular el número de filas y columnas
    n = int(np.sqrt(num_blocks))
    rows = cols = n

    # Calcular tamaños de bloques
    block_width = width // cols
    block_height = height // rows

    blocks = []
    positions = []

    for i in range(rows):
        for j in range(cols):
            # Coordenadas del bloque
            left = j * block_width
            top = i * block_height
            right = left + block_width
            bottom = top + block_height

            # Extraer y transformar el bloque
            block = image.crop((left, top, right, bottom))
            blocks.append(block)
            positions.append((i, j))

    return blocks, positions, (rows, cols)

def preprocess_image(image, analyze_blocks=False, num_blocks=9):
    """Preprocesa una imagen para la entrada del modelo."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if analyze_blocks:
        blocks, positions, grid_size = split_image_blocks(image, num_blocks)
        processed_blocks = []
        for block in blocks:
            # Aplicar transformaciones a cada bloque
            processed_block = transform(block).unsqueeze(0)
            processed_blocks.append(processed_block)
        return processed_blocks, positions, grid_size
    else:
        # Proceso normal para imagen completa
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)

def load_images_from_folder(folder_path):
    """Carga y preprocesa imágenes desde carpetas para entrenamiento."""
    images = []
    labels = []
    class_names = []

    class_folders = sorted([d for d in os.listdir(folder_path) 
                          if os.path.isdir(os.path.join(folder_path, d))])

    total_images = 0
    for idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        class_names.append(class_folder)

        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]

        print(f"Procesando clase {class_folder}: {len(image_files)} imágenes")

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image_tensor = transform(image)

                if image_tensor.shape == (3, 224, 224):
                    images.append(image_tensor)
                    labels.append(idx)
                    total_images += 1
                else:
                    print(f"Dimensiones incorrectas en {image_path}: {image_tensor.shape}")
            except Exception as e:
                print(f"Error procesando {image_path}: {str(e)}")

    X_train = torch.stack(images)
    X_train = X_train.numpy()
    y_train = np.array(labels)

    return X_train, y_train, class_names