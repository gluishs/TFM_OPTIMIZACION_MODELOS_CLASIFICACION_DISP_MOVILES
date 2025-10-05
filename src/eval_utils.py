import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import torch

def visualizar_curvas(history):
    """
    Muestra las curvas de pérdida (loss) y exactitud (accuracy) 
    de entrenamiento y validación a lo largo de las épocas.

    Parámetros:
        history (dict): diccionario con las claves
                        'train_loss', 'val_loss', 
                        'train_acc', 'val_acc'.
    
    Salida:
        Gráficos de matplotlib mostrando las curvas.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss durante el entrenamiento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy durante el entrenamiento')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



def evaluar_modelo(model, dataloader, clases, device):
    """
    Evalúa el modelo sobre un dataloader y reporta métricas de clasificación
    (precision, recall, F1 por clase y macro/weighted). También genera la
    matriz de confusión para analizar errores por clase.

    Parámetros
    ----------
    model : torch.nn.Module
        Modelo ya cargado en `device` y en modo eval en la función.
    dataloader : torch.utils.data.DataLoader
        Conjunto de evaluación (y_true/y_pred se derivan aquí).
    clases : list[str]
        Nombres legibles de las clases para los ejes y el reporte.
    device : torch.device
        'cpu' o 'cuda'.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("📊 Clasification Report:")
    print(classification_report(all_labels, all_preds, target_names=clases))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=clases, yticklabels=clases, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.show()


def medir_inferencia(modelo, dataloader, device):
    """
    Mide el tiempo promedio de inferencia por imagen 
    utilizando el modelo en modo evaluación.

    Parámetros:
        modelo (torch.nn.Module): red entrenada.
        dataloader (torch.utils.data.DataLoader): datos de prueba.
        device (torch.device): 'cpu' o 'cuda'.
    
    Retorno:
        float: tiempo promedio de inferencia por imagen en segundos.
    """
    modelo.eval()
    tiempos = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Tomar solo una imagen por lote
            input_img = inputs[0].unsqueeze(0)

            inicio = time.perf_counter()
            _ = modelo(input_img)
            fin = time.perf_counter()

            tiempos.append(fin - inicio)

    tiempo_promedio = sum(tiempos) / len(tiempos)
    print(f"⏱️ Tiempo de inferencia promedio por imagen: {tiempo_promedio * 1000:.3f} ms")
    return tiempo_promedio
