# DiPreSi
**Diagnóstico y Predicción de Síntomas por Sequía y Enfermedades en Plantas de Banano mediante Modelación Matemática a partir de Huellas Espectrales**

## 🎯 Objetivo Global

Este repositorio contiene el desarrollo de un sistema de diagnóstico y predicción para identificar síntomas de estrés hídrico y enfermedades en plantas de banano Williams (variedad Cavendish) mediante el análisis de sus firmas espectrales obtenidas por espectroscopía Vis-NIR (Visible-Near Infrared).

## 📋 Descripción del Proyecto

El proyecto utiliza técnicas de aprendizaje automático y análisis espectral para clasificar plantas de banano según su estado de salud, identificando específicamente:

- **Estado de salud general**: Plantas sanas vs. plantas con estrés
- **Tipo de estrés**: Estrés hídrico vs. Enfermedades fúngicas/bacterianas
- **Tipo de enfermedad**: Ralstonia (bacteria) vs. Fusarium (hongo)

### 🔬 Metodología

El análisis se basa en datos espectrales recolectados mediante un espectrómetro Vis-NIR que mide la reflectancia de las hojas en el rango de 350-2500 nm. Este rango incluye:

- **Rango Visible y UV-A (350-780 nm)**: Información sobre pigmentos como la clorofila
- **Rango NIR (780-2500 nm)**: Información sobre estructura celular interna y composición química (agua, celulosa, proteínas)

### 🌱 Tratamientos Experimentales

El dataset incluye los siguientes tratamientos de estrés aplicados a las plantas:

- **Control**: Plantas sanas sin estrés (línea base)
- **Ralstonia**: Infección bacteriana (marchitez bacteriana)
- **Fusarium**: Infección fúngica 
- **E_Hidrico**: Estrés hídrico (sequía)
- **Tratamientos Combinados**: 
  - Ral_Fus: Infección simultánea de Ralstonia y Fusarium
  - Fus_EH: Fusarium + Estrés hídrico
  - Y otros


## 🔍 Análisis Inicial

### 1. Exploración de Datos Espectrales (`visualizacion.ipynb`)

Este notebook contiene el análisis exploratorio de las firmas espectrales:

#### Conceptos Clave:
- **Espectrofotometría**: Medición de cómo la luz interactúa con las hojas
- **Reflectancia**: Fracción de luz reflejada (valores de 0 a 1)
- **Huella espectral**: Patrón único de reflectancia que caracteriza el estado fisiológico

#### Hallazgos Principales:

1. **Estructura de los Datos**:
   - 2151 columnas de longitudes de onda (350-2500 nm)
   - Grupos de 30 plantas por tratamiento
   - Datos organizados en múltiples hojas de Excel

2. **Análisis de Regiones Espectrales**:
   
   **Región NIR Plateau (700-1378 nm)**:
   - Alta reflectancia correlacionada con estructura celular robusta
   - Diferencias significativas entre tratamientos
   - Absorción característica alrededor de 1000 nm (enlaces químicos específicos)
   
   **Región Química (1400-2500 nm)**:
   - Sensible a contenido de agua (bandas O-H)
   - Absorción por celulosa y lignina (enlaces C-H)
   - **Anomalía Fus_EH**: Pico distintivo en 1750-1871 nm que sugiere alteraciones químicas profundas

3. **Comparación Entre Tratamientos**:
   - Control vs. Ralstonia: Divergencia en 700-1100 nm (estructura y agua)
   - Fus_EH muestra comportamiento único en región 1500-2500 nm
   - Patrones espectrales permiten discriminación entre condiciones

4. **Visualización Interactiva**:
   - Herramienta desarrollada para explorar espectros por planta y tratamiento
   - Identificación de longitudes de onda con máxima diferenciación
   - Análisis de variabilidad intra e inter-grupo

## 🤖 Modelos de Clasificación

### Modelos Implementados (`Models_Cassification.ipynb`)

El notebook desarrolla **tres clasificadores binarios** utilizando PCA para reducción de dimensionalidad:

#### 1. **Clasificación: Plantas Sanas vs. Enfermas**

**Objetivo**: Identificar si una planta está bajo estrés o es saludable

**Modelos**:
- **Regresión Logística** + PCA
  - Hiperparámetros: penalty (L1, L2, elasticnet), C, solver
  - Búsqueda exhaustiva con GridSearchCV
  
- **SVM** + PCA
  - Kernels: linear, rbf, poly, sigmoid
  - Optimización de C, gamma, degree

**Datos**: Dataset completo excluyendo tratamiento Fus_EH

#### 2. **Clasificación: Estrés Hídrico vs. Enfermedad Fúngica/Bacteriana**

**Objetivo**: Distinguir entre estrés abiótico (sequía) y biótico (patógenos)

**Clases**:
- Hydric_Stress (E_Hidrico)
- Fungus_Disease (Ralstonia + Fusarium)

**Modelos**: Regresión Logística + PCA con optimización de hiperparámetros

#### 3. **Clasificación: Ralstonia vs. Fusarium**

**Objetivo**: Diferenciar entre infección bacteriana y fúngica

**Modelos**:
- Regresión Logística + PCA
- SVM + PCA
- Bagging Decision Tree + PCA
  - Meta-estimador con múltiples árboles de decisión
  - Bootstrap sampling para reducir varianza

### 🔧 Pipeline de Modelado

1. **Preprocesamiento**:
   - Extracción de características espectrales (columnas 350-2500 nm)
   - División train/test (80/10-20%, estratificada)

2. **Reducción de Dimensionalidad**:
   - PCA con 5-40 componentes principales
   - Retiene información relevante, reduce overfitting

3. **Optimización**:
   - GridSearchCV con validación cruzada (5-fold)
   - Métricas: accuracy, balanced accuracy
   - Búsqueda de mejores hiperparámetros

4. **Evaluación**:
   - Matriz de confusión
   - Classification report (precision, recall, f1-score)
   - Accuracy en conjunto de prueba

## 📊 Resultados y Métricas

Cada modelo se evalúa con:
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los casos positivos reales, cuántos se detectan
- **F1-Score**: Media armónica de precision y recall
- **Confusion Matrix**: Visualización de predicciones vs. realidad

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Análisis de Datos**: pandas, numpy
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Visualización**: matplotlib, seaborn, plotly
- **Reducción Dimensionalidad**: PCA, t-SNE
- **Notebooks**: Jupyter

## 📦 Dependencias Principales

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
plotly
ipywidgets
openpyxl  # Para leer archivos Excel
```

## 🚀 Próximos Pasos

- [ ] Análisis de importancia de longitudes de onda específicas
- [ ] Implementación de modelos de clasificación multiclase
- [ ] Desarrollo de modelos de regresión para severidad del estrés
- [ ] Validación con datos de campo
- [ ] Despliegue de modelo como herramienta de diagnóstico

## 👥 Contribuciones

Este proyecto es parte de una investigación académica sobre diagnóstico temprano de estrés en cultivos de banano mediante técnicas no destructivas de espectroscopía.

## 📄 Licencia

[Especificar licencia del proyecto]

---

**Nota**: Los datos espectrales provienen de experimentos controlados en plantas de banano Williams sometidas a diferentes condiciones de estrés para estudiar sus respuestas fisiológicas y desarrollar herramientas de diagnóstico predictivo.
