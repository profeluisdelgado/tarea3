# Telescopio MLOps: Observación de Sirio & Simulador de Data Drift 🌌

Este repositorio contiene una aplicación web interactiva desarrollada con **Streamlit** y **Pillow** diseñada para simular la puesta en producción de un modelo clasificador astronómico y monitorear la degradación de sus predicciones frente al **Data Drift (Deriva de Datos)**.

El sistema toma como referencia datos del **Sloan Digital Sky Survey (SDSS)** y permite al usuario interactuar en tiempo real con las variables espectroscópicas de la estrella **Sirio** y desestabilizar la distribución global del universo de referencia.

---

## 🎯 Características Principales

### 1. Simulador de Predicciones Individuales
* **Controles Interactivos**: Modifica las coordenadas (`alpha` y `delta`), las magnitudes en cinco bandas fotométricas (`u`, `g`, `r`, `i`, `z`) y el corrimiento al rojo (`redshift`).
* **Visualización Dinámica (Pillow)**:
  * El desplazamiento en coordenadas altera el encuadre en el buscador del telescopio mostrando una retícula de puntería (verde cuando está alineado, roja cuando hay error de apuntamiento).
  - Los filtros fotométricos modifican en tiempo real los canales RGB de la imagen `Sirio.jpg`.
  - El corrimiento al rojo simula un desplazamiento cromático (reddening) progresivo hacia el canal rojo.
* **Inferencia en Tiempo Real**: Consume un modelo `RandomForestClassifier` que clasifica inmediatamente el objeto entre Estrella (`STAR`), Galaxia (`GALAXY`) o Quásar (`QSO`), junto con su barra de probabilidades de confianza.

### 2. Laboratorio de Data Drift & Métricas MLOps
* **Modo de Perturbación**: Permite simular fallas en los motores del telescopio, contaminación lumínica/nubosidad en la atmósfera y sesgos espectrales en los sensores.
* **Cálculos Estadísticos Nativos**:
  * **Population Stability Index (PSI)** para medir cuantitativamente el desplazamiento de las distribuciones.
  * **Test de Kolmogorov-Smirnov (KS-test)** para verificar significancia estadística de la deriva de datos.
* **Semáforo MLOps Inteligente**: Alerta visual con código de colores (Verde = Seguro, Amarillo = Advertencia, Rojo = Peligro Crítico) que reacciona si el PSI acumulado supera los umbrales estándar de MLOps (PSI > 0.25).
* **Gráficos Dinámicos (Plotly)**: Visualizador interactivo de histogramas y densidades comparando la distribución original (Referencia) contra la alterada.

---

## 📂 Estructura del Proyecto

```text
├── app.py                  # Código principal del dashboard Streamlit
├── generar_recursos.py     # Script para entrenar el modelo y generar el CSV de base
├── dataset_referencia.csv  # Dataset de referencia astronómica (1500 registros)
├── modelo_final.pkl        # Modelo RandomForest serializado
├── Sirio.jpg               # Imagen de referencia astronómica
├── requirements.txt        # Dependencias de Python del proyecto
└── README.md               # Este archivo de documentación
```

---

## 🛠️ Instalación y Configuración Local

Sigue estos pasos para clonar el repositorio, configurar tu entorno y ejecutar la aplicación en tu computadora local:

### 1. Clonar el repositorio
```bash
git clone https://github.com/profeluisdelgado/tarea3.git
cd tarea3
```
*Nota: Si estás ejecutando la aplicación localmente en el directorio de trabajo del asistente, la ruta de entrada para tu equipo es:*
```powershell
cd "C:\Users\Luis Delgado\.gemini\antigravity\scratch\mlops-drift-app"
```

### 2. Crear y activar un entorno virtual

**En Windows (CMD - Símbolo del Sistema):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En macOS/Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Generar el modelo y datos de prueba
Si deseas entrenar el modelo clasificador desde cero y crear el archivo de referencia de datos astronómicos, ejecuta:
```bash
python generar_recursos.py
```

### 5. Iniciar la aplicación
Ejecuta el servidor web local de Streamlit:
```bash
streamlit run app.py
```

Abre tu navegador de preferencia en la dirección que arroje la consola (usualmente **`http://localhost:8501`**).

---

## 📊 MLOps: Interpretación del Semáforo y PSI
El **Population Stability Index (PSI)** se evalúa bajo las siguientes reglas de la industria:
* **PSI < 0.1 (Estable - Verde 🟢)**: La distribución actual es consistente con la de entrenamiento. El modelo es confiable.
* **0.1 ≤ PSI < 0.25 (Advertencia - Amarillo 🟡)**: El modelo empieza a experimentar drift. Es recomendable planificar un reentrenamiento.
* **PSI ≥ 0.25 (Drift Crítico - Rojo 🔴)**: Desplazamiento severo en las variables de entrada. El modelo se considera degradado; sus predicciones no son seguras ni válidas técnicamente.
