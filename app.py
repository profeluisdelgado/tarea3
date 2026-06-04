import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Configuración de página con diseño ancho y título moderno
st.set_page_config(
    page_title="Telescopio MLOps - Simulador de Data Drift",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyectar CSS personalizado para estética premium, modo oscuro espacial y semáforo brillante
st.markdown("""
<style>
    /* Estilos del Dashboard Premium */
    .main {
        background-color: #0b0e14;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0b0e14;
    }
    
    /* Tarjetas y Contenedores */
    .metric-card {
        background-color: #151a24;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    
    .status-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    
    /* Títulos e interfaces */
    h1, h2, h3 {
        color: #63b3ed !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Semáforo Visual Moderno y Radiante */
    .semaphore-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        background-color: #1e2530;
        padding: 15px;
        border-radius: 50px;
        width: fit-content;
        margin: 15px auto;
        border: 1px solid #3b4252;
    }
    
    .light {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background-color: #2b333f;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    
    .light.green.active {
        background-color: #10b981;
        box-shadow: 0 0 15px #10b981, inset 0 0 5px rgba(255,255,255,0.4);
    }
    .light.yellow.active {
        background-color: #f59e0b;
        box-shadow: 0 0 15px #f59e0b, inset 0 0 5px rgba(255,255,255,0.4);
    }
    .light.red.active {
        background-color: #ef4444;
        box-shadow: 0 0 15px #ef4444, inset 0 0 5px rgba(255,255,255,0.4);
    }
    
    /* Centrado e interactividad del panel lateral */
    [data-testid="stSidebar"] {
        background-color: #0f131a;
        border-right: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. CARGA DE RECURSOS (DATOS Y MODELO)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_model_and_data():
    """
    Carga el modelo clasificador pre-entrenado y el dataset de referencia.
    Si no existen, muestra una advertencia útil.
    """
    model_path = "modelo_final.pkl"
    dataset_path = "dataset_referencia.csv"
    
    model = None
    df_ref = None
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        st.error(f"Error: No se encontró el modelo en '{model_path}'. Ejecuta generar_recursos.py primero.")
        
    if os.path.exists(dataset_path):
        df_ref = pd.read_csv(dataset_path)
    else:
        st.error(f"Error: No se encontró el dataset de referencia en '{dataset_path}'. Ejecuta generar_recursos.py primero.")
        
    return model, df_ref

model, df_ref = load_model_and_data()

# ------------------------------------------------------------------------------
# 2. MANIPULACIÓN DE IMAGEN CON PILLOW
# ------------------------------------------------------------------------------

def modify_sirio_image(image_path, alpha, delta, u, g, r, i, z, redshift):
    """
    Modifica dinámicamente la imagen de Sirio.jpg basada en coordenadas,
    magnitudes fotométricas de filtros SDSS y corrimiento al rojo (redshift).
    """
    if not os.path.exists(image_path):
        # Crear imagen negra de fallback si no existe la de Sirio
        fallback = Image.new("RGB", (300, 300), (10, 15, 25))
        draw = ImageDraw.Draw(fallback)
        draw.text((80, 140), "Sirio.jpg no encontrada", fill=(100, 150, 200))
        return fallback

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # --- A. Apuntamiento (alpha, delta) ---
    # Sirio real: RA = 101.287, Dec = -16.716
    alpha_ref = 101.287
    delta_ref = -16.716
    
    # Desplazamiento amplificado en píxeles (sensibilidad 25px por grado)
    dx = int((alpha - alpha_ref) * 25)
    dy = int((delta - delta_ref) * 25)
    
    # Asegurar que el recorte no cause desborde
    dx = max(-width // 3, min(width // 3, dx))
    dy = max(-height // 3, min(height // 3, dy))
    
    # Crear un lienzo negro y pegar la estrella desplazada
    canvas = Image.new("RGB", (width, height), (5, 5, 10))
    canvas.paste(img, (-dx, -dy))
    
    # --- B. Efectos de magnitudes (u, g, r, i, z) ---
    arr = np.array(canvas, dtype=np.float32)
    
    # Fórmulas de flujo simulado (magnitud menor = estrella más brillante)
    # Magnitud de referencia en SDSS de 20.0
    weight_u = 10 ** ((20.0 - u) / 4.0)  # Ultravioleta -> Afecta canal Azul
    weight_g = 10 ** ((20.0 - g) / 4.0)  # Verde -> Afecta canal Verde
    weight_r = 10 ** ((20.0 - r) / 4.0)  # Rojo -> Afecta canal Rojo
    
    # Bandas infrarrojas (i y z) aumentan la ganancia/brillo general de fondo
    weight_ir = 10 ** ((20.0 - (i + z) / 2.0) / 4.0)
    
    # Ajustar canales RGB
    total_weights = (weight_u + weight_g + weight_r) / 3.0
    if total_weights > 0:
        gain_r = weight_r / total_weights
        gain_g = weight_g / total_weights
        gain_b = weight_u / total_weights
    else:
        gain_r = gain_g = gain_b = 1.0
        
    arr[:, :, 0] *= gain_r * weight_ir
    arr[:, :, 1] *= gain_g * weight_ir
    arr[:, :, 2] *= gain_b * weight_ir
    
    # --- C. Corrimiento al rojo (redshift) ---
    if redshift > 0:
        # Desplazamiento espectral hacia el rojo (atenúa azul/verde, potencia rojo)
        shift_factor = min(redshift / 3.0, 0.95)
        r_chan = arr[:, :, 0].copy()
        g_chan = arr[:, :, 1].copy()
        b_chan = arr[:, :, 2].copy()
        
        arr[:, :, 0] = r_chan + shift_factor * (g_chan + b_chan) * 0.6
        arr[:, :, 1] = g_chan * (1.0 - shift_factor)
        arr[:, :, 2] = b_chan * (1.0 - shift_factor)
    elif redshift < 0:
        # Corrimiento al azul (blueshift)
        shift_factor = min(abs(redshift) * 10, 0.9)
        r_chan = arr[:, :, 0].copy()
        g_chan = arr[:, :, 1].copy()
        b_chan = arr[:, :, 2].copy()
        
        arr[:, :, 2] = b_chan + shift_factor * (r_chan + g_chan) * 0.6
        arr[:, :, 0] = r_chan * (1.0 - shift_factor)
        arr[:, :, 1] = g_chan * (1.0 - shift_factor)
        
    # Recortar valores de píxeles
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    processed_img = Image.fromarray(arr)
    
    # --- D. Desenfoque por contaminación / magnitudes tenues ---
    # Si la estrella es muy tenue, simula mala atmósfera difuminando
    mean_mag = (u + g + r) / 3.0
    if mean_mag > 21.0:
        blur_val = (mean_mag - 21.0) * 1.5
        processed_img = processed_img.filter(ImageFilter.GaussianBlur(blur_val))
        
    # --- E. Dibujar retícula del telescopio (Crosshair de centrado) ---
    draw = ImageDraw.Draw(processed_img)
    cx, cy = width // 2, height // 2
    # Dibujar líneas del crosshair con transparencia simulada usando color rojo oscuro o verde
    # Usaremos un color rojo de alerta si la estrella está desalineada
    is_aligned = (abs(dx) < 10 and abs(dy) < 10)
    line_color = (16, 185, 129) if is_aligned else (239, 68, 68)  # Verde si alineado, Rojo si no
    
    draw.line((cx - 25, cy, cx + 25, cy), fill=line_color, width=1)
    draw.line((cx, cy - 25, cx, cy + 25), fill=line_color, width=1)
    draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=line_color, width=1)
    
    return processed_img

# ------------------------------------------------------------------------------
# 3. CÁLCULO ESTADÍSTICO DE DATA DRIFT
# ------------------------------------------------------------------------------

def calculate_psi(expected, actual, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) entre la distribución esperada (original)
    y la distribución actual (desplazada por el villano).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Definir los cuantiles basados en el dataset de referencia
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins) # Eliminar bordes repetidos si el rango es estrecho
    
    if len(bins) < 2:
        return 0.0
        
    # Ajustar bordes a infinito para cubrir cualquier valor del dataset desplazado
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calcular conteos en bins
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Calcular porcentajes
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Evitar divisiones por cero y log(0)
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    
    # Ecuación del PSI: sum( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value

def calculate_drift_metrics(reference_df, shifted_df):
    """
    Compara columna a columna el dataset de referencia con el alterado.
    Calcula la diferencia de medias, KS-test y PSI.
    """
    metrics = []
    numeric_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
    
    for col in numeric_cols:
        ref_data = reference_df[col].values
        shift_data = shifted_df[col].values
        
        ref_mean = np.mean(ref_data)
        shift_mean = np.mean(shift_data)
        
        # Test Kolmogorov-Smirnov
        ks_stat, p_value = stats.ks_2samp(ref_data, shift_data)
        
        # Population Stability Index
        psi_val = calculate_psi(ref_data, shift_data)
        
        # Estado de drift por columna
        if psi_val < 0.1:
            status = "🟢 Estable"
        elif psi_val < 0.25:
            status = "🟡 Drift Moderado"
        else:
            status = "🔴 Drift Crítico"
            
        metrics.append({
            'Variable': col,
            'Media Ref.': f"{ref_mean:.3f}",
            'Media Alt.': f"{shift_mean:.3f}",
            'Diferencia %': f"{((shift_mean - ref_mean) / (ref_mean if ref_mean != 0 else 1) * 100):+.1f}%",
            'p-valor (KS)': f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}",
            'PSI': f"{psi_val:.4f}",
            'Estado': status,
            'psi_num': psi_val
        })
        
    return pd.DataFrame(metrics)

# ------------------------------------------------------------------------------
# 4. RENDERIZACIÓN DE PESTAÑAS DE LA UI
# ------------------------------------------------------------------------------

# Cabecera principal del Dashboard
st.markdown("<h1 style='text-align: center;'>🌌 Telescopio MLOps: Observación de Sirio & Data Drift</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 1.1em;'>Consola MLOps interactiva para evaluar el impacto de las variaciones espectrales y espaciales en la clasificación estelar.</p>", unsafe_allow_html=True)

# Verificar carga correcta
if model is None or df_ref is None:
    st.warning("⚠️ Carga de recursos fallida. Asegúrate de ejecutar `generar_recursos.py` primero.")
else:
    # Crear las dos pestañas críticas requeridas
    tab1, tab2 = st.tabs(["🎯 Predicciones Individuales", "⚡ Laboratorio de Data Drift (Simulador de Estrés)"])

    # ==========================================================================
    # PESTAÑA 1: SIMULADOR DE PREDICCIONES INDIVIDUALES
    # ==========================================================================
    with tab1:
        st.subheader("Simulador de Parámetros de Apuntamiento y Espectroscopía")
        
        # Crear layout de tres columnas
        col_controls, col_prediction, col_image = st.columns([2, 1, 2])
        
        with col_controls:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🎛️ Controles del Objeto</h4>", unsafe_allow_html=True)
            
            # Controles de coordenadas (Apuntamiento)
            st.markdown("##### Posición en el Cielo (Coordenadas)")
            alpha = st.slider("Alpha (RA - Ascensión Recta)", 0.0, 360.0, 101.287, step=0.01, help="Ángulo coordenado ecuatorial horizontal (Sirio está en 101.287°)")
            delta = st.slider("Delta (Dec - Declinación)", -90.0, 90.0, -16.716, step=0.01, help="Ángulo coordenado ecuatorial vertical (Sirio está en -16.716°)")
            
            # Controles de magnitudes SDSS
            st.markdown("##### Magnitudes de Filtros SDSS (u, g, r, i, z)")
            st.caption("Nota: Magnitud menor = estrella más brillante espectralmente.")
            u = st.slider("Magnitud u (Ultravioleta)", 10.0, 30.0, 15.8, step=0.1)
            g = st.slider("Magnitud g (Verde)", 10.0, 30.0, 14.5, step=0.1)
            r = st.slider("Magnitud r (Rojo)", 10.0, 30.0, 14.2, step=0.1)
            i = st.slider("Magnitud i (Infrarrojo Cercano)", 10.0, 30.0, 14.0, step=0.1)
            z = st.slider("Magnitud z (Infrarrojo Lejano)", 10.0, 30.0, 13.9, step=0.1)
            
            # Control de redshift
            st.markdown("##### Parámetro Cosmológico")
            redshift = st.slider("Redshift (z)", min_value=-0.05, max_value=5.00, value=0.000, step=0.001, format="%.4f", help="Corrimiento al rojo cosmológico. Valores > 0 indican alejamiento acelerado.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Ejecutar inferencia inmediata
        input_data = pd.DataFrame([{
            'alpha': alpha, 'delta': delta,
            'u': u, 'g': g, 'r': r, 'i': i, 'z': z,
            'redshift': redshift
        }])
        
        pred_class = model.predict(input_data)[0]
        pred_probs = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        with col_prediction:
            st.markdown("<div class='metric-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<h4>🔮 Predicción MLOps</h4>", unsafe_allow_html=True)
            
            # Mostrar la predicción destacada
            color_map = {"STAR": "#10b981", "GALAXY": "#63b3ed", "QSO": "#a855f7"}
            badge_color = color_map.get(pred_class, "#e2e8f0")
            
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; border-radius: 8px; background-color: #1e2530; margin-bottom: 20px; border-top: 4px solid {badge_color};'>
                <p style='margin: 0; font-size: 0.9em; color: #a0aec0; text-transform: uppercase;'>Objeto Detectado</p>
                <h2 style='margin: 5px 0 0 0; color: {badge_color} !important; font-size: 2.2em;'>{pred_class}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar barras de confianza para cada clase
            st.markdown("##### Confianza del Clasificador:")
            for cls, prob in zip(classes, pred_probs):
                col_c, col_p = st.columns([1, 4])
                col_c.write(f"**{cls}**")
                col_p.progress(float(prob))
                st.caption(f"Probabilidad: {prob * 100:.2f}%")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_image:
            st.markdown("<div class='metric-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<h4>🔭 Visualización en el Ocular (Sirio.jpg)</h4>", unsafe_allow_html=True)
            
            # Modificar y mostrar la imagen reactivamente
            modified_img = modify_sirio_image("Sirio.jpg", alpha, delta, u, g, r, i, z, redshift)
            st.image(modified_img, width='stretch', caption=f"Sirio simulada - Filtros activos (RA: {alpha:.2f}°, Dec: {delta:.2f}°, z: {redshift:.3f})")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # PESTAÑA 2: LABORATORIO E INTERFAZ DE DATA DRIFT (SIMULADOR DE ESTRÉS)
    # ==========================================================================
    with tab2:
        st.subheader("Módulo de Inestabilidad Global")
        st.write("Modifica el comportamiento macro del universo para evaluar cómo el modelo responde ante cambios de distribución drásticos.")
        
        # Configurar las perturbaciones del villano
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### 🚗 Desplazamiento de Coordenadas (Apuntamiento)", unsafe_allow_html=True)
            drift_coord_shift = st.slider("Falla del Motor de Montura (°)", -5.0, 5.0, 0.0, step=0.1, help="Simula una descalibración física en los motores del telescopio que desplaza sistemáticamente la posición reportada de las estrellas.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_v2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### 🌫️ Atenuación de Magnitud (Atmosfera/Nubes)", unsafe_allow_html=True)
            drift_mag_shift = st.slider("Incremento de Magnitudes (Atenuación)", 0.0, 4.0, 0.0, step=0.1, help="Simula nubes o contaminación lumínica severa. Hace que todos los objetos parezcan más tenues (magnitud más alta) en todas las bandas simultáneamente.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_v3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### 🚀 Aceleración Cósmica (Desplazamiento Espectral)", unsafe_allow_html=True)
            drift_redshift_shift = st.slider("Corrimiento al Rojo Promedio (+z)", 0.00, 2.00, 0.00, step=0.05, help="Simula un sesgo cosmológico o falla en el sensor espectral que introduce un corrimiento sistemático de longitud de onda hacia el rojo en toda la muestra.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Generar dataset alterado (Drifted) a partir del original
        df_shifted = df_ref.copy()
        
        # Aplicar perturbaciones
        df_shifted['alpha'] += drift_coord_shift
        df_shifted['delta'] += drift_coord_shift
        df_shifted['u'] += drift_mag_shift
        df_shifted['g'] += drift_mag_shift
        df_shifted['r'] += drift_mag_shift
        df_shifted['i'] += drift_mag_shift
        df_shifted['z'] += drift_mag_shift
        df_shifted['redshift'] += drift_redshift_shift
        
        # Calcular métricas de Data Drift
        drift_results = calculate_drift_metrics(df_ref, df_shifted)
        max_psi = drift_results['psi_num'].max()
        
        # Determinar el color del semáforo reactivo
        green_class = "green"
        yellow_class = "yellow"
        red_class = "red"
        
        status_text = ""
        status_bg = ""
        
        if max_psi < 0.1:
            green_class += " active"
            status_text = "SEGURO - Sin desajuste de distribución detectable. Las predicciones son altamente confiables."
            status_bg = "rgba(16, 185, 129, 0.15)"
            status_border = "#10b981"
        elif max_psi < 0.25:
            yellow_class += " active"
            status_text = "ADVERTENCIA - Se detecta Data Drift moderado. Monitoree de cerca la precisión de las predicciones."
            status_bg = "rgba(245, 158, 11, 0.15)"
            status_border = "#f59e0b"
        else:
            red_class += " active"
            status_text = "PELIGRO CRÍTICO - Data Drift severo. Las predicciones ya no cumplen los criterios éticos ni técnicos de fiabilidad."
            status_bg = "rgba(239, 68, 68, 0.15)"
            status_border = "#ef4444"
            
        # Renderizar semáforo visual brillante
        col_status, col_semaforo = st.columns([3, 1])
        
        with col_status:
            st.markdown(f"""
            <div class='status-card' style='background-color: {status_bg}; border: 1px solid {status_border};'>
                <h3 style='margin:0 0 10px 0; color: {status_border} !important;'>Indicador de Salud del Modelo</h3>
                <p style='font-size: 1.15em; margin: 0; font-weight: bold;'>{status_text}</p>
                <p style='margin: 8px 0 0 0; font-size: 0.9em; color: #a0aec0;'>PSI Máximo Registrado: <b>{max_psi:.4f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_semaforo:
            st.markdown(f"""
            <div style='text-align: center;'>
                <span style='font-size: 0.9em; color: #a0aec0; text-transform: uppercase; font-weight: bold;'>Semáforo MLOps</span>
                <div class="semaphore-container">
                    <div class="light {green_class}"></div>
                    <div class="light {yellow_class}"></div>
                    <div class="light {red_class}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Mostrar tabla de métricas detallada
        st.markdown("#### 📊 Desglose de Métricas de Drift por Variable")
        st.dataframe(
            drift_results[['Variable', 'Media Ref.', 'Media Alt.', 'Diferencia %', 'p-valor (KS)', 'PSI', 'Estado']],
            width='stretch',
            hide_index=True
        )
        
        # Gráficas de densidad comparativas
        st.markdown("#### 📈 Visualizador Espectral de Distribuciones (Original vs Alterado)")
        
        available_vars = ['redshift', 'u', 'g', 'r', 'i', 'z', 'alpha', 'delta']
        selected_var = st.selectbox("Selecciona la variable a analizar:", available_vars, index=0)
        
        # Crear gráfico con Plotly
        fig = go.Figure()
        
        # Datos para graficar
        ref_series = df_ref[selected_var].values
        shifted_series = df_shifted[selected_var].values
        
        # Agregar histograma para Datos de Referencia
        fig.add_trace(go.Histogram(
            x=ref_series,
            name="Referencia (Original)",
            opacity=0.6,
            marker_color='#10b981',
            nbinsx=40,
            histnorm='probability density'
        ))
        
        # Agregar histograma para Datos Alterados
        fig.add_trace(go.Histogram(
            x=shifted_series,
            name="Alterado (Drifted)",
            opacity=0.6,
            marker_color='#ef4444',
            nbinsx=40,
            histnorm='probability density'
        ))
        
        fig.update_layout(
            title=f"Distribución de la variable '{selected_var}' (Normalizada por densidad)",
            barmode='overlay',
            paper_bgcolor='#151a24',
            plot_bgcolor='#151a24',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#2d3748', title=selected_var),
            yaxis=dict(gridcolor='#2d3748', title="Densidad de Probabilidad"),
            legend=dict(x=0.8, y=0.9, bgcolor='rgba(21, 26, 36, 0.8)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mensajes educativos de MLOps
        st.markdown("""
        > [!TIP]
        > **¿Cómo interpretar el PSI (Population Stability Index)?**
        > - **PSI < 0.1 (Estable)**: La distribución de las variables actuales es idéntica a la distribución con la que se entrenó el modelo. No hay peligro.
        > - **0.1 ≤ PSI < 0.25 (Alerta)**: Hay un leve corrimiento de datos. Las predicciones del modelo pueden comenzar a perder precisión. Es recomendable reentrenar pronto.
        > - **PSI ≥ 0.25 (Drift Crítico)**: Las variables han sufrido un cambio drástico. Las suposiciones del modelo de clasificación ya no son válidas y las predicciones no son seguras.
        """)
