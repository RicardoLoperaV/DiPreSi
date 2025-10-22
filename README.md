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


**Nota**: Los datos espectrales provienen de experimentos controlados en plantas de banano Williams sometidas a diferentes condiciones de estrés para estudiar sus respuestas fisiológicas y desarrollar herramientas de diagnóstico predictivo.
