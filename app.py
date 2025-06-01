import os
import json
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
import cv2
# from PIL import Image # No es estrictamente necesario si usamos cv2 para todo
import io
import base64

# Para características avanzadas (si se usan las mismas que en Colab)
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage.measure import regionprops, label
from skimage.morphology import disk, opening, closing
from scipy import ndimage


app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
MODELS_BASE_PATH = 'models' # Ruta a la carpeta de modelos
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_BASE_PATH, exist_ok=True) # Crear si no existe

# Variables globales para modelos
modelo_cnn_global = None
modelo_arbol_decision_global = None
sistema_experto_global = None
feature_scaler_global = None
mapeo_clases_global = None
feature_names_list_global = None

# --- Definición de la clase SistemaExpertoAvanzado directamente aquí ---
class SistemaExpertoAvanzado:
    def __init__(self, feature_names): # Ahora espera feature_names
        self.reglas = []
        self.feature_names = feature_names # Nombres de las características en el orden esperado
        self.inicializar_reglas_avanzadas()

    def inicializar_reglas_avanzadas(self):
        # Estas reglas deben ser consistentes con las características que realmente extraes
        # y con los feature_names cargados.
        self.reglas.append({
            'nombre': 'Black Rot',
            'condiciones': {
                'black_pixel_ratio_min': 0.05, 'rgb_std_0_min': 40,
                'contrast_mean_min': 0.2, 'edge_density_canny_min': 0.08,
            }, 'confianza': 0.80
        })
        self.reglas.append({
            'nombre': 'Esca (Black Measles)',
            'condiciones': {
                'esca_pixel_ratio_min': 0.03, 'hsv_mean_0_min': 15,
                'hsv_mean_0_max': 35, 'homogeneity_mean_max': 0.7,
                'num_regions_min': 15,
            }, 'confianza': 0.75
        })
        self.reglas.append({
            'nombre': 'Leaf Blight (Isariopsis Leaf Spot)',
            'condiciones': {
                'blight_pixel_ratio_min': 0.02, 'rgb_mean_0_min': 100,
                'dissimilarity_mean_min': 0.3, 'gabor_0_min': 0.05,
            }, 'confianza': 0.70
        })
        self.reglas.append({
            'nombre': 'Healthy',
            'condiciones': {
                'healthy_pixel_ratio_min': 0.35, 'rgb_mean_1_min': 90,
                'hsv_mean_0_min': 38, 'hsv_mean_0_max': 85,
                'homogeneity_mean_min': 0.6, 'num_regions_max': 40
            }, 'confianza': 0.85
        })

    def evaluar_condiciones(self, features_dict, condiciones_regla):
        for param_key, valor_limite in condiciones_regla.items():
            feature_name_base = param_key.rsplit('_', 1)[0] if param_key.endswith(('_min', '_max')) else param_key
            if feature_name_base not in features_dict:
                return False
            valor_actual = features_dict[feature_name_base]
            if param_key.endswith('_min') and valor_actual < valor_limite: return False
            if param_key.endswith('_max') and valor_actual > valor_limite: return False
        return True

    def diagnosticar(self, features_list_ordered):
        if len(features_list_ordered) != len(self.feature_names):
            return [{'enfermedad': 'Error en Features', 'confianza': 0.1, 'regla_activada': 'Input Mismatch'}]
        features_dict = dict(zip(self.feature_names, features_list_ordered))
        resultados = []
        for regla in self.reglas:
            if self.evaluar_condiciones(features_dict, regla['condiciones']):
                resultados.append({'enfermedad': regla['nombre'], 'confianza': regla['confianza'], 'regla_activada': regla['nombre']})
        if not resultados:
            default_disease = 'Indeterminado'
            default_confidence = 0.2
            if features_dict.get('healthy_pixel_ratio', 0) > 0.3: default_disease, default_confidence = 'Healthy', 0.4
            elif features_dict.get('black_pixel_ratio', 0) > 0.03: default_disease, default_confidence = 'Black Rot', 0.35
            resultados.append({'enfermedad': default_disease, 'confianza': default_confidence, 'regla_activada': 'default_fallback'})
        resultados.sort(key=lambda x: x['confianza'], reverse=True)
        return resultados
# --- Fin de la definición de SistemaExpertoAvanzado ---

def cargar_todos_los_modelos():
    global modelo_cnn_global, modelo_arbol_decision_global, sistema_experto_global, \
           feature_scaler_global, mapeo_clases_global, feature_names_list_global
    
    models_loaded_status = {'cnn': False, 'tree': False, 'expert': False, 'scaler': False, 'mapeo': False, 'features_names': False}

    try:
        # Cargar Nombres de Características PRIMERO (necesario para el Sistema Experto)
        features_path = os.path.join(MODELS_BASE_PATH, "feature_names_list.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_names_list_global = json.load(f)
            print("✓ Nombres de características cargados.")
            models_loaded_status['features_names'] = True
        else:
            print(f"⚠️ Archivo de nombres de características no encontrado en {features_path}. El sistema experto y árbol de decisión pueden no funcionar correctamente.")
            # Usar una lista de fallback si es necesario para que el sistema experto se inicialice
            feature_names_list_global = ['rgb_mean_0', 'hsv_std_2', 'black_pixel_ratio', 'esca_pixel_ratio', 'blight_pixel_ratio', 'healthy_pixel_ratio', 'contrast_mean', 'homogeneity_mean', 'energy_mean', 'correlation_mean', 'lbp_0', 'gabor_0', 'edge_density_canny', 'opening_diff', 'mean_area', 'num_regions', 'spectral_energy'] # Ejemplo, ajustar a tus features reales
            print("Usando lista de nombres de características de fallback.")


        # Cargar Sistema Experto (ahora depende de feature_names_list_global)
        sistema_experto_path = os.path.join(MODELS_BASE_PATH, "sistema_experto_obj.pkl")
        if os.path.exists(sistema_experto_path):
             with open(sistema_experto_path, 'rb') as f:
                sistema_experto_global = pickle.load(f)
             print("✓ Sistema Experto cargado desde archivo.")
             models_loaded_status['expert'] = True
        elif feature_names_list_global: # Si no hay pkl pero sí feature_names, inicializarlo
            sistema_experto_global = SistemaExpertoAvanzado(feature_names_list_global)
            print("✓ Sistema Experto inicializado (sin .pkl, usando clase y feature_names).")
            models_loaded_status['expert'] = True
        else:
            print(f"⚠️ Sistema Experto no pudo ser cargado ni inicializado.")


        # Cargar Modelo CNN
        cnn_path = os.path.join(MODELS_BASE_PATH, "modelo_cnn_mejorado.keras")
        if os.path.exists(cnn_path):
            modelo_cnn_global = load_model(cnn_path)
            print("✓ Modelo CNN cargado.")
            models_loaded_status['cnn'] = True
        else:
            print(f"⚠️ Modelo CNN no encontrado en {cnn_path}.")

        # Cargar Árbol de Decisión
        tree_path = os.path.join(MODELS_BASE_PATH, "modelo_arbol_decision.pkl")
        if os.path.exists(tree_path):
            with open(tree_path, 'rb') as f:
                modelo_arbol_decision_global = pickle.load(f)
            print("✓ Árbol de Decisión cargado.")
            models_loaded_status['tree'] = True
        else:
            print(f"⚠️ Árbol de Decisión no encontrado en {tree_path}.")

        # Cargar Scaler
        scaler_path = os.path.join(MODELS_BASE_PATH, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                feature_scaler_global = pickle.load(f)
            print("✓ Scaler de características cargado.")
            models_loaded_status['scaler'] = True
        else:
            print(f"⚠️ Scaler no encontrado en {scaler_path}.")

        # Cargar Mapeo de Clases
        mapeo_path = os.path.join(MODELS_BASE_PATH, "mapeo_clases.json")
        if os.path.exists(mapeo_path):
            with open(mapeo_path, 'r') as f:
                mapeo_clases_global = json.load(f)
            # Asegurar que las claves sean enteros si es necesario (depende de cómo se guardó)
            mapeo_clases_global = {int(k) if k.isdigit() else k: v for k, v in mapeo_clases_global.items()}
            print("✓ Mapeo de Clases cargado.")
            models_loaded_status['mapeo'] = True
        else:
            print(f"⚠️ Mapeo de Clases no encontrado en {mapeo_path}. Usando mapeo de fallback.")
            mapeo_clases_global = {0: 'Black Rot', 1: 'Esca (Black Measles)', 2: 'Leaf Blight (Isariopsis Leaf Spot)', 3: 'Healthy'}


        if not all(models_loaded_status.values()):
            print("\nADVERTENCIA: No todos los modelos y artefactos se cargaron correctamente. El sistema podría funcionar en modo simulación para algunos componentes.")
        
        return models_loaded_status

    except Exception as e:
        print(f"Error crítico durante la carga de modelos: {e}")
        print("El sistema podría no funcionar como se espera.")
        return models_loaded_status


def analisis_avanzado_pixeles_local(img_array_rgb, target_size=(256, 256)):
    """
    Función para extraer características, similar a la de Colab.
    """
    try:
        img_rgb = cv2.resize(img_array_rgb, target_size)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB) # LAB requiere BGR input para cvtColor usualmente

        # Stats de Color
        color_stats = {
            'rgb_mean': np.mean(img_rgb, axis=(0, 1)), 'rgb_std': np.std(img_rgb, axis=(0, 1)),
            'hsv_mean': np.mean(img_hsv, axis=(0, 1)), 'hsv_std': np.std(img_hsv, axis=(0, 1)),
            'lab_mean': np.mean(img_lab, axis=(0, 1)), 'lab_std': np.std(img_lab, axis=(0, 1)),
        }
        
        # Ratios de píxeles
        black_pixels = np.sum((img_rgb[:,:,0] < 50) & (img_rgb[:,:,1] < 50) & (img_rgb[:,:,2] < 50))
        black_ratio = black_pixels / (img_rgb.shape[0] * img_rgb.shape[1])
        esca_mask = ((img_hsv[:,:,0] >= 10) & (img_hsv[:,:,0] <= 30) & (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,2] >= 100) & (img_hsv[:,:,2] <= 200))
        esca_ratio = np.sum(esca_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        blight_mask = ((img_hsv[:,:,0] <= 20) & (img_hsv[:,:,1] >= 100) & (img_hsv[:,:,2] >= 50) & (img_hsv[:,:,2] <= 150))
        blight_ratio = np.sum(blight_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        healthy_mask = ((img_hsv[:,:,0] >= 40) & (img_hsv[:,:,0] <= 80) & (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,2] >= 50))
        healthy_ratio = np.sum(healthy_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        
        # GLCM
        glcm_features = {}
        try:
            glcm = graycomatrix(img_gray, distances=[1, 3], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
            glcm_props_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            for prop in glcm_props_list:
                glcm_features[f'{prop}_mean'] = np.mean(graycoprops(glcm, prop))
        except Exception: # Fallback si GLCM falla
             for prop in glcm_props_list: glcm_features[f'{prop}_mean'] = 0.0


        # LBP
        radius = 3; n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
        
        # Gabor
        gabor_responses = []
        for theta_deg in [0, 45, 90, 135]:
            for frequency in [0.1, 0.5]:
                try:
                    real, _ = gabor(img_gray, frequency=frequency, theta=np.radians(theta_deg))
                    gabor_responses.extend([np.mean(np.abs(real)), np.std(real)])
                except Exception: # Fallback
                     gabor_responses.extend([0.0, 0.0])
        
        # Bordes y Morfología
        edges_canny = cv2.Canny(img_gray, 50, 150)
        edge_density_canny = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
        kernel = disk(3)
        img_opened = opening(img_gray, kernel)
        opening_diff = np.mean(np.abs(img_gray.astype(float) - img_opened.astype(float)))
        
        # Regiones
        segments = slic(img_rgb, n_segments=30, compactness=10, sigma=1, start_label=1)
        regions = regionprops(segments, intensity_image=img_gray)
        region_areas = [r.area for r in regions]
        region_stats = {'mean_area': np.mean(region_areas) if region_areas else 0, 'num_regions': len(regions)}

        # Espectral
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        spectral_energy = np.sum(magnitude_spectrum**2)
        
        features_dict = {
            **{f'rgb_mean_{i}': color_stats['rgb_mean'][i] for i in range(3)},
            **{f'rgb_std_{i}': color_stats['rgb_std'][i] for i in range(3)},
            **{f'hsv_mean_{i}': color_stats['hsv_mean'][i] for i in range(3)},
            **{f'hsv_std_{i}': color_stats['hsv_std'][i] for i in range(3)},
            **{f'lab_mean_{i}': color_stats['lab_mean'][i] for i in range(3)},
            **{f'lab_std_{i}': color_stats['lab_std'][i] for i in range(3)},
            'black_pixel_ratio': black_ratio, 'esca_pixel_ratio': esca_ratio,
            'blight_pixel_ratio': blight_ratio, 'healthy_pixel_ratio': healthy_ratio,
            **glcm_features,
            **{f'lbp_{i}': lbp_hist[i] for i in range(min(10, len(lbp_hist)))},
            **{f'gabor_{i}': gabor_responses[i] for i in range(min(10, len(gabor_responses)))},
            'edge_density_canny': edge_density_canny, 'opening_diff': opening_diff,
            **region_stats, 'spectral_energy': spectral_energy
            # Asegúrate de incluir TODAS las características que tus modelos esperan
            # Si faltan, tus modelos (especialmente el árbol y el scaler) fallarán.
        }
        
        # Ordenar características según feature_names_list_global
        if feature_names_list_global:
            ordered_feature_values = [features_dict.get(name, 0.0) for name in feature_names_list_global] # Usar 0.0 para las faltantes
        else: # Fallback si no hay nombres de features
            ordered_feature_values = list(features_dict.values()) 
            print("Advertencia: Usando orden de características por defecto, podría no coincidir con los modelos.")
            
        return features_dict, np.array([ordered_feature_values])

    except Exception as e:
        print(f"Error en analisis_avanzado_pixeles_local: {e}")
        # Devolver un diccionario y array de ceros con la forma esperada si feature_names_list_global está disponible
        if feature_names_list_global:
            dummy_features_dict = {name: 0.0 for name in feature_names_list_global}
            dummy_features_array = np.zeros((1, len(feature_names_list_global)))
            return dummy_features_dict, dummy_features_array
        return {}, np.array([[]]) # Peor caso


def generar_recomendaciones_api(enfermedad):
    recomendaciones = {
        "Black Rot": ["Eliminar partes infectadas.", "Aplicar fungicida de cobre.", "Mejorar ventilación."],
        "Esca (Black Measles)": ["Poda sanitaria.", "Proteger heridas de poda.", "Manejo nutricional."],
        "Leaf Blight (Isariopsis Leaf Spot)": ["Fungicidas preventivos.", "Mejorar drenaje.", "Eliminar restos infectados."],
        "Healthy": ["Continuar buenas prácticas.", "Monitoreo regular."]
    }
    return recomendaciones.get(enfermedad, ["Consultar a un agrónomo para recomendaciones específicas."])

def sistema_integrado_api_local(img_array_rgb):
    if not mapeo_clases_global or not feature_names_list_global:
        return {'error': 'El sistema no está completamente inicializado (mapeo o nombres de features faltantes).'}

    features_dict, features_array_ordered = analisis_avanzado_pixeles_local(img_array_rgb)

    if features_array_ordered.size == 0 : # Si la extracción falló completamente
         return {'error': 'Fallo en la extracción de características de la imagen.'}


    # CNN
    clase_idx_cnn, confianza_cnn, pred_probs_cnn = -1, 0.0, np.zeros(len(mapeo_clases_global))
    if modelo_cnn_global:
        img_cnn_resized = cv2.resize(img_array_rgb, (224, 224)) # Tamaño esperado por EfficientNetB0
        img_cnn_array = img_to_array(img_cnn_resized)
        img_cnn_array_expanded = np.expand_dims(img_cnn_array, axis=0)
        img_cnn_processed = efficientnet_preprocess_input(img_cnn_array_expanded)
        pred_probs_cnn = modelo_cnn_global.predict(img_cnn_processed, verbose=0)[0]
        clase_idx_cnn = np.argmax(pred_probs_cnn)
        confianza_cnn = float(pred_probs_cnn[clase_idx_cnn])
    else: # Simulación si CNN no está cargada
        clase_idx_cnn = np.random.randint(0,len(mapeo_clases_global))
        confianza_cnn = np.random.rand() * 0.3 + 0.6 # Confianza simulada alta
        pred_probs_cnn = np.random.rand(len(mapeo_clases_global))
        pred_probs_cnn = pred_probs_cnn / np.sum(pred_probs_cnn)
        print("Simulando predicción CNN.")


    # Árbol de Decisión
    clase_idx_dt, confianza_dt, pred_probs_dt = -1, 0.0, np.zeros(len(mapeo_clases_global))
    if modelo_arbol_decision_global and feature_scaler_global:
        # Asegurarse que features_array_ordered tenga la forma correcta (1, num_features)
        if features_array_ordered.ndim == 1:
            features_array_ordered_dt = features_array_ordered.reshape(1, -1)
        else:
            features_array_ordered_dt = features_array_ordered

        if features_array_ordered_dt.shape[1] == feature_scaler_global.n_features_in_:
            features_scaled = feature_scaler_global.transform(features_array_ordered_dt)
            pred_probs_dt = modelo_arbol_decision_global.predict_proba(features_scaled)[0]
            clase_idx_dt = np.argmax(pred_probs_dt)
            confianza_dt = float(pred_probs_dt[clase_idx_dt])
        else:
            print(f"Discrepancia en número de features para DT. Esperado: {feature_scaler_global.n_features_in_}, Obtenido: {features_array_ordered_dt.shape[1]}. Simulando DT.")
            clase_idx_dt = np.random.randint(0,len(mapeo_clases_global))
            confianza_dt = np.random.rand() * 0.3 + 0.5 # Confianza simulada media
            pred_probs_dt[clase_idx_dt] = confianza_dt
            # Llenar el resto para que sume (aproximadamente)
            pred_probs_dt_others = np.random.rand(len(mapeo_clases_global)-1)*(1-confianza_dt) if len(mapeo_clases_global)>1 else np.array([])
            j=0
            for i in range(len(mapeo_clases_global)):
                if i != clase_idx_dt and j < len(pred_probs_dt_others):
                    pred_probs_dt[i] = pred_probs_dt_others[j]
                    j+=1
            pred_probs_dt = pred_probs_dt / np.sum(pred_probs_dt)


    elif modelo_arbol_decision_global and not feature_scaler_global:
        print("Scaler no cargado, usando features no escaladas para DT (puede ser impreciso).")
        pred_probs_dt = modelo_arbol_decision_global.predict_proba(features_array_ordered)[0] # Asume que features_array_ordered ya tiene forma (1, N)
        clase_idx_dt = np.argmax(pred_probs_dt)
        confianza_dt = float(pred_probs_dt[clase_idx_dt])
    else: # Simulación si DT o scaler no están cargados
        clase_idx_dt = np.random.randint(0,len(mapeo_clases_global))
        confianza_dt = np.random.rand() * 0.3 + 0.5 # Confianza simulada media
        pred_probs_dt[clase_idx_dt] = confianza_dt
        pred_probs_dt_others = np.random.rand(len(mapeo_clases_global)-1)*(1-confianza_dt) if len(mapeo_clases_global)>1 else np.array([])
        j=0
        for i in range(len(mapeo_clases_global)):
            if i != clase_idx_dt and j < len(pred_probs_dt_others):
                pred_probs_dt[i] = pred_probs_dt_others[j]
                j+=1
        pred_probs_dt = pred_probs_dt / np.sum(pred_probs_dt)
        print("Simulando predicción Árbol de Decisión.")

    # Sistema Experto
    clase_nombre_es, confianza_es, clase_idx_es = "Indeterminado", 0.0, -1
    if sistema_experto_global:
        # El sistema experto espera una lista simple de valores de características
        diagnostico_es_list = sistema_experto_global.diagnosticar(features_array_ordered.flatten().tolist())
        if diagnostico_es_list:
            diagnostico_es_principal = diagnostico_es_list[0]
            clase_nombre_es = diagnostico_es_principal['enfermedad']
            confianza_es = float(diagnostico_es_principal['confianza'])
            for idx, nombre_map in mapeo_clases_global.items():
                if nombre_map.lower() in clase_nombre_es.lower():
                    clase_idx_es = idx
                    break
    else:
        print("Simulando predicción Sistema Experto.")
        clase_idx_es = np.random.randint(0,len(mapeo_clases_global))
        confianza_es = np.random.rand() * 0.3 + 0.4 # Confianza simulada baja-media
        clase_nombre_es = mapeo_clases_global.get(clase_idx_es, "Simulado")


    # Votación ponderada
    votos = np.zeros(len(mapeo_clases_global))
    peso_cnn = 0.6 * (1 + confianza_cnn) if modelo_cnn_global else 0
    peso_dt = 0.25 * (1 + confianza_dt) if modelo_arbol_decision_global else 0
    peso_es = 0.15 * (1 + confianza_es) if sistema_experto_global else 0.1 # Darle un peso mínimo si es el único

    total_peso = peso_cnn + peso_dt + peso_es
    if total_peso == 0: total_peso = 1 # Evitar división por cero

    votos += pred_probs_cnn * (peso_cnn / total_peso)
    if clase_idx_dt != -1: votos[clase_idx_dt] += (confianza_dt * peso_dt / total_peso) # O usar pred_probs_dt
    if clase_idx_es != -1: votos[clase_idx_es] += (confianza_es * peso_es / total_peso)

    clase_final_idx = np.argmax(votos)
    confianza_final_score = float(votos[clase_final_idx])
    
    recomendaciones = generar_recomendaciones_api(mapeo_clases_global.get(clase_final_idx, "Desconocido"))

    return {
        'clase_final': mapeo_clases_global.get(clase_final_idx, "Indeterminado"),
        'confianza_final': confianza_final_score,
        'resultados_individuales': {
            'cnn': {'clase': mapeo_clases_global.get(clase_idx_cnn, "N/A"), 'confianza': confianza_cnn, 
                    'probabilidades': {mapeo_clases_global[i]: float(pred_probs_cnn[i]) for i in range(len(mapeo_clases_global))}},
            'arbol': {'clase': mapeo_clases_global.get(clase_idx_dt, "N/A"), 'confianza': confianza_dt,
                      'probabilidades': {mapeo_clases_global[i]: float(pred_probs_dt[i]) for i in range(len(mapeo_clases_global))}},
            'experto': {'clase': clase_nombre_es, 'confianza': confianza_es}
        },
        'caracteristicas_clave': {k: features_dict.get(k, 0.0) for k in ['healthy_pixel_ratio', 'black_pixel_ratio', 'edge_density_canny', 'num_regions'][:4]},
        'recomendaciones': recomendaciones
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analizar', methods=['POST'])
def analizar_imagen_endpoint():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        imagen_bytes = archivo.read()
        nparr = np.frombuffer(imagen_bytes, np.uint8)
        img_cv2_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv2_bgr is None:
            return jsonify({'error': 'No se pudo decodificar la imagen. Asegúrate de que sea un formato válido (JPG, PNG).'}), 400
        
        img_cv2_rgb = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB) # Convertir a RGB para consistencia
        
        resultado_completo = sistema_integrado_api_local(img_cv2_rgb)
        return jsonify(resultado_completo)

    except Exception as e:
        print(f"Error en /analizar: {e}")
        return jsonify({'error': f'Ocurrió un error durante el análisis: {str(e)}'}), 500

# --- Inicialización ---
if __name__ == '__main__':
    print("Iniciando aplicación Flask GrapeAI...")
    print("Cargando modelos y artefactos...")
    status_carga = cargar_todos_los_modelos()
    print(f"Estado de carga de modelos: {status_carga}")
    
    if not mapeo_clases_global:
        print("¡ADVERTENCIA CRÍTICA! El mapeo de clases no se cargó. La aplicación puede no funcionar.")
    if not feature_names_list_global:
        print("¡ADVERTENCIA CRÍTICA! La lista de nombres de características no se cargó. Modelos como DT y ES pueden fallar.")

    app.run(debug=True, host='0.0.0.0', port=5000)