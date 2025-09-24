# Optimización de Modelos de Clasificación para Dispositivos Móviles con Recursos Limitados

> Trabajo Fin de Máster (VIU, 2025). Investigación aplicada para equilibrar **precisión y eficiencia** en clasificación de imágenes ejecutada **on-device** en móviles y sistemas embebidos.



## 📌 Descripción
Este proyecto evalúa técnicas de optimización para llevar modelos de **aprendizaje profundo** a dispositivos con **memoria, cómputo y energía limitados**. Se compara un **Teacher** de alta capacidad (**Vision Transformer – ViT Base**) frente a un **Student** eficiente (**MobileNetV3 Small**), entrenado en modo clásico y con **Knowledge Distillation (KD)**.  
El estudio se realiza sobre **Food-101** (101 clases, >100k imágenes), un escenario multiclase exigente y representativo para despliegue móvil



## 🎯 Objetivos
- Investigar la viabilidad de clasificación multiclase **on-device** manteniendo precisión competitiva.
- Entrenar un **Teacher (ViT Base)** como referencia y un **Student (MobileNetV3 Small)** en dos variantes: **clásico** y **con KD**.
- Comparar **precisión**, **tamaño del modelo**, **latencia** (proyección a móvil) y **uso de memoria**.
- Demostrar que **KD** mejora la **generalización** del Student manteniendo un tamaño reducido (≤10 MB; ideal ≤5 MB).



## ⚙️ Metodología
- **Dataset**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)  
- **Modelos**:
  - **Teacher**: ViT Base (gran capacidad, guía del conocimiento).
  - **Student**: MobileNetV3 Small (eficiente para móviles).
- **Entrenamientos**:
  1) Student **clásico**.  
  2) Student **con KD** (logits del Teacher con temperatura T y mezcla α).
- **Evaluación**:
  - Métricas de clasificación (Accuracy, F1 macro), **matriz de confusión** y análisis por clase.
  - **Tamaño del modelo** (MB) y **latencia** (medición en PC + proyección a móvil por gama).
  - Discusión de compromisos **precisión ↔ eficiencia** para despliegue real.



## 📊 Resultados principales

| Modelo                         | Rol      | Tamaño aprox. | Accuracy test | Macro F1 | Hallazgos clave |
|--------------------------------|----------|---------------|---------------|----------|-----------------|
| **ViT Base**                   | Teacher  | ~327 MB       | **0.97**      | **0.97** | Modelo de referencia con altísima precisión. Sin embargo, su gran tamaño y coste computacional lo hacen inviable para móviles. |
| **MobileNetV3 Small**          | Student  | ~6.3 MB       | **0.73**      | **0.73** | Modelo ligero y eficiente, pero con claras limitaciones de generalización en un escenario multiclase complejo como Food-101. |
| **MobileNetV3 Small + KD**     | Student  | ~6.3 MB       | **0.90**      | **0.89** | La Knowledge Distillation mejora sustancialmente el rendimiento del Student, alcanzando un nivel cercano al Teacher con tamaño reducido y latencias bajas. |

> Resumen: **KD logra un salto del 73% al 90% en accuracy**, validando su eficacia para optimizar modelos ligeros sin incrementar el tamaño, lo que lo hace adecuado para **dispositivos móviles de gama media y baja**.



## 🧪 Reproducibilidad

### 1) Requisitos
- Python 3.10+
- PyTorch + torchvision
- timm, scikit-learn, numpy, matplotlib, tqdm, yaml

Instalación rápida:
```bash
pip install -r requirements.txt
