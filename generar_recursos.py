import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generar_datos_y_modelo():
    # Configurar semilla para reproducibilidad astronómica
    np.random.seed(42)
    
    n_samples = 1500
    n_each = n_samples // 3  # Balanceado: 500 estrellas, 500 galaxias, 500 QSOs
    
    # Generar coordenadas uniformes en el cielo
    # alpha: Ascensión Recta (0° a 360°)
    alpha = np.random.uniform(0.0, 360.0, n_samples)
    # delta: Declinación (-90° a 90°)
    delta = np.random.uniform(-90.0, 90.0, n_samples)
    
    # Inicializar arreglos para magnitudes fotométricas y redshift
    u = np.zeros(n_samples)
    g = np.zeros(n_samples)
    r = np.zeros(n_samples)
    i = np.zeros(n_samples)
    z = np.zeros(n_samples)
    redshift = np.zeros(n_samples)
    clases = []
    
    # --- Generar Estrellas (STAR) ---
    # Las estrellas locales tienen redshift cercano a cero y magnitudes más brillantes (menor valor numérico)
    for idx in range(0, n_each):
        base_mag = np.random.uniform(14.0, 19.5)
        u[idx] = base_mag + np.random.normal(1.2, 0.15)
        g[idx] = base_mag + np.random.normal(0.4, 0.1)
        r[idx] = base_mag + np.random.normal(0.0, 0.05)
        i[idx] = base_mag + np.random.normal(-0.2, 0.08)
        z[idx] = base_mag + np.random.normal(-0.3, 0.1)
        redshift[idx] = np.random.normal(0.0, 0.0004) # Muy bajo
        clases.append("STAR")
        
    # --- Generar Galaxias (GALAXY) ---
    # Galaxias muestran corrimiento al rojo moderado y son ligeramente más opacas
    for idx in range(n_each, 2 * n_each):
        base_mag = np.random.uniform(16.0, 21.5)
        u[idx] = base_mag + np.random.normal(1.8, 0.2)
        g[idx] = base_mag + np.random.normal(0.9, 0.1)
        r[idx] = base_mag + np.random.normal(0.3, 0.08)
        i[idx] = base_mag + np.random.normal(0.0, 0.08)
        z[idx] = base_mag + np.random.normal(-0.1, 0.1)
        redshift[idx] = np.random.uniform(0.01, 0.8) # Moderado
        clases.append("GALAXY")
        
    # --- Generar Quásares (QSO) ---
    # Quásares son objetos sumamente lejanos (alto redshift) y su color es característico
    for idx in range(2 * n_each, n_samples):
        base_mag = np.random.uniform(18.5, 23.0)
        u[idx] = base_mag + np.random.normal(0.5, 0.2)
        g[idx] = base_mag + np.random.normal(0.3, 0.1)
        r[idx] = base_mag + np.random.normal(0.2, 0.08)
        i[idx] = base_mag + np.random.normal(0.1, 0.08)
        z[idx] = base_mag + np.random.normal(0.0, 0.1)
        redshift[idx] = np.random.uniform(0.8, 4.5) # Muy lejano
        clases.append("QSO")
        
    # Crear DataFrame
    df = pd.DataFrame({
        'alpha': alpha,
        'delta': delta,
        'u': u,
        'g': g,
        'r': r,
        'i': i,
        'z': z,
        'redshift': redshift,
        'class': clases
    })
    
    # Guardar a CSV
    csv_path = "C:/Users/Luis Delgado/.gemini/antigravity/scratch/mlops-drift-app/dataset_referencia.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset de referencia guardado en: {csv_path}")
    
    # Separar variables y objetivo
    X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']]
    y = df['class']
    
    # Partición para validación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entrenar clasificador
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Modelo entrenado. Exactitud en Entrenamiento: {train_acc:.4f} | Test: {test_acc:.4f}")
    
    # Serializar modelo
    model_path = "C:/Users/Luis Delgado/.gemini/antigravity/scratch/mlops-drift-app/modelo_final.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo serializado guardado en: {model_path}")

if __name__ == "__main__":
    generar_datos_y_modelo()
