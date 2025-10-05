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

## 📌 Conclusiones

- **ViT Base (Teacher)**: ofrece la mayor precisión (97%), pero su tamaño (327 MB) y latencia lo hacen inviable en móviles.  
- **MobileNetV3 Student clásico**: ligero (6 MB) y rápido, pero con baja generalización (73% de accuracy).  
- **MobileNetV3 Student + KD**: logra un equilibrio ideal (90% accuracy, tamaño reducido y baja latencia), validando la **Knowledge Distillation** como técnica clave para **optimizar modelos ligeros en dispositivos con recursos limitados**.  

En conclusión, el modelo **MobileNetV3 con KD** es la opción más adecuada para **despliegue real en móviles**, combinando eficiencia y precisión sin comprometer memoria ni tiempo de inferencia.  


## 📖 Referencias
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*.  
- Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*.  
- Howard, A., et al. (2019). *Searching for MobileNetV3*.  
- Zong, X., Zhang, Y., Wang, L., & Lin, S. (2023). *Edge AI Benchmarking of MobileNetV3 on Smartphones*. *Journal of Systems Architecture*.  


## 📜 Licencia
Este proyecto se distribuye bajo la licencia MIT.  
Consulta el archivo [LICENSE](LICENSE) para más detalles.  


## ✉️ Contacto
Autor: **Luis Guaman**  
Repositorio: [TFM_OPTIMIZACION_MODELOS_CLASIFICACION_DISP_MOVILES](https://github.com/gluishs/TFM_OPTIMIZACION_MODELOS_CLASIFICACION_DISP_MOVILES)

## 🧪 Reproducibilidad

### 1) Requisitos
- Python 3.10+
- PyTorch + torchvision
- timm, scikit-learn, numpy, matplotlib, tqdm, yaml
- 
## 🚀 Uso del repositorio

Este repositorio **no está orientado a instalación como paquete**, sino como proyecto académico/documental del TFM.  
Su propósito es **explorar, reproducir y analizar** entrenamientos y resultados.  

- Para revisar los experimentos → abrir los notebooks en `notebooks/`.  
- Para ver modelos ya entrenados → carpeta `models/`.  
- Para consultar resultados de métricas y gráficas → carpeta `results/`.  
- Para revisar código auxiliar y funciones → carpeta `src/`.


## 📂 Estructura de directorios

```bash
├── history/                  # Historiales de entrenamiento (curvas, logs, métricas)
├── models/                   # Modelos entrenados (.pth)
├── notebooks/                # Jupyter Notebooks con el desarrollo del proyecto
├── results/                  # Resultados de evaluación y comparativas
│   ├── vit_base/             # Resultados del modelo Teacher (ViT Base)
│   ├── student_classic/      # Resultados del modelo Student clásico (MobileNetV3 Small)
│   ├── student_kd/           # Resultados del modelo Student con Knowledge Distillation
│   └── comparison/           # Comparativas globales entre modelos
├── src/                      # Código auxiliar y scripts de soporte
└── splits_food101_80_15_5.json  # División del dataset Food-101 (train/val/test)

