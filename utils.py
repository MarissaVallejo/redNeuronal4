# Importaciones básicas primero
import numpy as np

# Importaciones de visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones de scikit-learn
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """Visualiza el historial de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de precisión
    ax1.plot(history['train_acc'])
    ax1.plot(history['val_acc'])
    ax1.set_title('Precisión del modelo')
    ax1.set_ylabel('Precisión')
    ax1.set_xlabel('Época')
    ax1.legend(['Entrenamiento', 'Validación'])
    
    # Gráfico de pérdida
    ax2.plot(history['train_loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Pérdida del modelo')
    ax2.set_ylabel('Pérdida')
    ax2.set_xlabel('Época')
    ax2.legend(['Entrenamiento', 'Validación'])
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Genera una matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    
    return fig
