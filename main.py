import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
from model import create_model, train_model
from preprocessing import preprocess_image, load_images_from_folder
from utils import plot_training_history, plot_confusion_matrix
import io
import base64

# Configuración de la página
st.set_page_config(page_title="Clasificador de Imágenes CNN", layout="wide")

# Agregar estilos CSS con animaciones mejoradas y Google Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Animaciones base */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from { 
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes scaleIn {
    from {
        transform: scale(0.95);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

/* Estilos base */
* {
    font-family: 'Inter', sans-serif;
}

/* Bloques de predicción */
.prediction-block {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 12px;
    margin: 8px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

.prediction-block:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10;
}

/* Zoom container */
.zoom-container {
    position: fixed;
    display: none;
    background: white;
    border: 1px solid #ddd;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    z-index: 1000;
    padding: 10px;
    border-radius: 5px;
}

.zoom-image {
    max-width: 300px;
    max-height: 300px;
}

/* Leyenda de colores -  STYLES MOVED TO create_color_legend FUNCTION*/

/* Bloques unificados */
.unified-blocks {
    border-style: dashed;
    border-width: 2px;
}

/* Resto de estilos... */
</style>
""", unsafe_allow_html=True)

def get_color_for_class(class_name):
    """Retorna el color asociado a cada clase."""
    color_map = {
        'DIQUE': (255, 107, 107),
        'SPB': (78, 205, 196),
        'SPP': (69, 183, 209),
        'SSM': (150, 206, 180),
        'VM': (255, 190, 11),
        'VOLCANICO': (255, 0, 110)
    }
    return color_map.get(class_name, (204, 204, 204))

def create_color_legend(class_names):
    """Crea una leyenda de colores usando componentes de Streamlit."""
    st.sidebar.markdown("### Leyenda de Clases")

    for class_name in class_names:
        color = get_color_for_class(class_name)
        st.sidebar.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                margin: 5px 0;
                padding: 8px;
                border-radius: 4px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
                ">
                <div style="
                    width: 20px;
                    height: 20px;
                    margin-right: 10px;
                    border-radius: 4px;
                    background-color: rgb{color};
                    "></div>
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #333;
                    font-size: 14px;
                    ">{class_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_zoom_container():
    """Crea el contenedor para el zoom."""
    return """
    <div class="zoom-container">
        <img class="zoom-image" src="" alt="Zoom view">
    </div>
    """

def create_overlay_image(image, predictions, positions, grid_size, confidence_threshold=0.0):
    """Crea una imagen superpuesta con los bloques coloreados."""
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')

    rows, cols = grid_size
    block_width = image.width // cols
    block_height = image.height // rows

    for pred, pos in zip(predictions, positions):
        class_name, confidence = pred['class'], pred['confidence']

        # Saltar bloques por debajo del umbral de confianza
        if confidence < confidence_threshold:
            continue

        row, col = pos
        # Calcular coordenadas
        x0 = col * block_width
        y0 = row * block_height
        x1 = x0 + block_width
        y1 = y0 + block_height

        # Obtener color y crear versión semitransparente
        color = get_color_for_class(class_name)
        color_with_alpha = (*color, int(128 * confidence))

        # Dibujar rectángulo redondeado con borde suave
        draw.rectangle([x0, y0, x1, y1], 
                      fill=color_with_alpha,
                      outline=(*color, 255),
                      width=2)

        # Crear fondo semitransparente para el texto
        text = f"{class_name}\n{confidence:.1%}"
        font_size = min(block_width // 8, 16)  # Ajustar tamaño de fuente según el bloque
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calcular dimensiones del texto
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Dibujar fondo del texto con bordes redondeados
        padding = 6
        text_bg = [
            x0 + 8,
            y0 + 8,
            x0 + text_width + padding * 2 + 8,
            y0 + text_height + padding * 2 + 8
        ]

        # Fondo semitransparente con degradado
        for i in range(3):  # Crear efecto de sombra suave
            alpha = 160 - i * 40
            offset = i * 1
            draw.rectangle([
                text_bg[0] - offset,
                text_bg[1] - offset,
                text_bg[2] + offset,
                text_bg[3] + offset
            ], fill=(0, 0, 0, alpha))

        # Agregar texto con sombra suave
        text_x = x0 + 8 + padding
        text_y = y0 + 8 + padding

        # Sombra del texto
        draw.text((text_x+1, text_y+1), text, font=font, fill=(0, 0, 0, 180))
        # Texto principal
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    return overlay

def main():
    st.title("Clasificador de Imágenes CNN")

    if "prediction_mode" not in st.session_state:
        st.session_state.prediction_mode = "Predicción"

    # Sidebar para navegación
    st.sidebar.title("Opciones")
    prediction_mode = st.sidebar.radio(
        "Modo de predicción",
        ["Entrenamiento", "Predicción", "Predicción por Bloques", "Predicción por Video"]
    )
    st.session_state.prediction_mode = prediction_mode

    if prediction_mode == "Predicción por Bloques":
        st.header("Predicción por Bloques")

        if not os.path.exists('modelo_cnn.pth'):
            st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
            return

        # Opciones de configuración de bloques
        col1, col2 = st.columns(2)
        with col1:
            num_blocks = st.select_slider(
                "Número de bloques",
                options=[4, 9, 16, 25],
                value=9,
                help="Seleccione el número total de bloques para el análisis"
            )

        with col2:
            confidence_threshold = st.slider(
                "Umbral de confianza",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Mostrar solo bloques con confianza mayor al umbral seleccionado"
            )

        uploaded_file = st.file_uploader(
            "Cargar imagen para predicción", 
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file:
            # Crear contenedor para zoom
            st.markdown(create_zoom_container(), unsafe_allow_html=True)

            image = Image.open(uploaded_file)

            # Cargar modelo y nombres de clases
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load('modelo_cnn.pth')
            class_names = checkpoint['class_names']

            # Agregar leyenda de colores en el sidebar
            create_color_legend(class_names)

            model = create_model(len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            with torch.no_grad():
                # Procesar bloques
                processed_blocks, positions, grid_size = preprocess_image(
                    image, 
                    analyze_blocks=True,
                    num_blocks=num_blocks
                )

                # Almacenar predicciones
                predictions = []
                results = []

                for block, pos in zip(processed_blocks, positions):
                    block = block.to(device)
                    outputs = model(block)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)

                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0][pred_class].item()
                    class_name = class_names[pred_class]

                    predictions.append({
                        'class': class_name,
                        'confidence': confidence,
                        'probabilities': probabilities[0].cpu().numpy()
                    })

                    if confidence >= confidence_threshold:
                        results.append({
                            'Bloque': f'({pos[0]+1},{pos[1]+1})',
                            'Clase': class_name,
                            'Confianza': f'{confidence:.2%}'
                        })

                # Crear imagen con bloques superpuestos
                overlay = create_overlay_image(image, predictions, positions, grid_size, confidence_threshold)
                result_image = Image.alpha_composite(image.convert('RGBA'), overlay)

                # Mostrar imagen resultante
                st.image(result_image, use_container_width=True)

                # Mostrar tabla de resultados
                if results:
                    with st.expander("Ver detalles de predicciones", expanded=True):
                        st.table(pd.DataFrame(results))
                else:
                    st.warning("No se encontraron bloques que superen el umbral de confianza seleccionado.")

    elif prediction_mode == "Predicción":
        st.header("Predicción")

        if not os.path.exists('modelo_cnn.pth'):
            st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
            return

        uploaded_file = st.file_uploader("Cargar imagen para predicción", 
                                    type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            # Contenedor con animación de fade-in
            with st.container():
                image = Image.open(uploaded_file)

                # Cargar modelo y nombres de clases
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load('modelo_cnn.pth')
                class_names = checkpoint['class_names']

                model = create_model(len(class_names))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                with torch.no_grad():
                    # Análisis de imagen completa
                    st.image(image, caption="Imagen cargada", use_container_width=True)
                    processed_image = preprocess_image(image).to(device)
                    outputs = model(processed_image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)

                    st.subheader("Resultados de la predicción:")
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0][pred_class].item()

                    st.write(f"Clase predicha: {class_names[pred_class]}")
                    st.write(f"Confianza: {confidence:.2%}")

                    fig, ax = plt.subplots()
                    probs = probabilities[0].cpu().numpy()
                    sns.barplot(x=class_names, y=probs)
                    plt.xticks(rotation=45)
                    plt.title("Probabilidades por clase")
                    plt.xlabel("Clase")
                    plt.ylabel("Probabilidad")
                    plt.tight_layout()
                    st.pyplot(fig)

    elif prediction_mode == "Predicción por Video":
        st.header("Predicción por Video")
        if not os.path.exists('modelo_cnn.pth'):
            st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
            return

        # Opciones de configuración
        col1, col2 = st.columns(2)
        with col1:
            analyze_blocks = st.checkbox("Analizar por bloques", value=False)
            

        with col2:
            pass


        uploaded_file = st.file_uploader("Cargar video para predicción", type=['mp4', 'avi'])

        if uploaded_file:
            # Guardar el video temporalmente
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Abrir el video con OpenCV
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Slider para navegar por los frames
            frame_idx = st.slider("Seleccionar frame", 0, total_frames-1, 0)

            # Cargar modelo y nombres de clases
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load('modelo_cnn.pth')
            class_names = checkpoint['class_names']

            # Agregar leyenda de colores en el sidebar si está en modo bloques
            if analyze_blocks:
                create_color_legend(class_names)
                num_blocks = st.select_slider(
                    "Número de bloques",
                    options=[4, 9, 16, 25],
                    value=9,
                    help="Seleccione el número total de bloques para el análisis"
                )
                confidence_threshold = st.slider(
                    "Umbral de confianza",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Mostrar solo bloques con confianza mayor al umbral seleccionado"
                )

            model = create_model(len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Ir al frame seleccionado
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Crear una imagen PIL
                pil_image = Image.fromarray(frame_rgb)

                # Procesar imagen y hacer predicción
                with torch.no_grad():
                    if analyze_blocks:
                        # Crear contenedor para zoom
                        st.markdown(create_zoom_container(), unsafe_allow_html=True)

                        # Procesar bloques
                        processed_blocks, positions, grid_size = preprocess_image(
                            pil_image, 
                            analyze_blocks=True,
                            num_blocks=num_blocks
                        )

                        # Almacenar predicciones
                        predictions = []
                        results = []

                        for block, pos in zip(processed_blocks, positions):
                            block = block.to(device)
                            outputs = model(block)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)

                            pred_class = torch.argmax(probabilities).item()
                            confidence = probabilities[0][pred_class].item()
                            class_name = class_names[pred_class]

                            predictions.append({
                                'class': class_name,
                                'confidence': confidence,
                                'probabilities': probabilities[0].cpu().numpy()
                            })

                            if confidence >= confidence_threshold:
                                results.append({
                                    'Bloque': f'({pos[0]+1},{pos[1]+1})',
                                    'Clase': class_name,
                                    'Confianza': f'{confidence:.2%}'
                                })

                        # Crear imagen con bloques superpuestos
                        overlay = create_overlay_image(pil_image, predictions, positions, grid_size, confidence_threshold)
                        result_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)

                        # Mostrar imagen resultante
                        st.image(result_image, caption=f"Frame {frame_idx} (Análisis por bloques)", use_container_width=True)

                        # Mostrar tabla de resultados
                        if results:
                            with st.expander("Ver detalles de predicciones", expanded=True):
                                st.table(pd.DataFrame(results))
                        else:
                            st.warning("No se encontraron bloques que superen el umbral de confianza seleccionado.")
                    else:
                        # Análisis de frame completo
                        processed_image = preprocess_image(pil_image).to(device)
                        outputs = model(processed_image)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

                        with col2:
                            st.subheader("Predicción:")
                            pred_class = torch.argmax(probabilities).item()
                            confidence = probabilities[0][pred_class].item()

                            st.write(f"Clase predicha: {class_names[pred_class]}")
                            st.write(f"Confianza: {confidence:.2%}")

                            # Gráfico de barras de probabilidades
                            fig, ax = plt.subplots()
                            probs = probabilities[0].cpu().numpy()
                            sns.barplot(x=class_names, y=probs)
                            plt.xticks(rotation=45)
                            plt.title("Probabilidades por clase")
                            plt.xlabel("Clase")
                            plt.ylabel("Probabilidad")
                            plt.tight_layout()
                            st.pyplot(fig)

            cap.release()
            # Eliminar archivo temporal
            os.remove(temp_path)

    else:  # Modo Entrenamiento
        st.header("Entrenamiento del Modelo")

        # Sección de carga de datos
        st.subheader("1. Cargar Datos de Entrenamiento")
        data_folder = st.text_input("Carpeta de datos de entrenamiento:", "dataset")

        if st.button("Cargar y Entrenar"):
            if os.path.exists(data_folder):
                with st.spinner('Cargando imágenes...'):
                    X_train, y_train, class_names = load_images_from_folder(data_folder)
                    st.success(f"Datos cargados exitosamente. Imágenes: {len(X_train)}")

                    # Entrenamiento del modelo
                    st.subheader("2. Entrenamiento del Modelo")
                    model = create_model(len(class_names))

                    with st.spinner('Entrenando el modelo...'):
                        history = train_model(model, X_train, y_train)

                        # Guardar el modelo entrenado
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'class_names': class_names
                        }
                        torch.save(checkpoint, 'modelo_cnn.pth')

                        # Mostrar gráficas de entrenamiento
                        st.subheader("3. Resultados del Entrenamiento")
                        fig = plot_training_history(history)
                        st.pyplot(fig)

                        st.success("¡Entrenamiento completado exitosamente!")
            else:
                st.error(f"La carpeta {data_folder} no existe.")

if __name__ == "__main__":
    main()