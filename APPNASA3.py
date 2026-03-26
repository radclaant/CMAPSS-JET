# ==========================================
# APP PANEL DE CONTROL - NASA C-MAPSS v4.0
# VERSIÓN: CON MODELOS PRE-ENTRENADOS INTEGRADOS
# ==========================================
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="NASA Turbofan C-MAPSS", page_icon="🚀", layout="wide")

# ═════════════════════════════════════════════════════════════════════════════════
# 1. RUTAS Y CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PKLS = os.path.join(BASE_DIR, 'pkls')

# Mapeo de FD00X a dataset_id (para cargar los archivos pkl correctos)
FD_MAPPING = {
    "FD001 - Nivel del Mar (HPC)": "FD001",
    "FD002 - 6 Condiciones (HPC)": "FD002",
    "FD003 - Nivel del Mar (HPC + Fan)": "FD003",
    "FD004 - 6 Condiciones (HPC + Fan)": "FD004",
}

# ═════════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES DE CARGA DE MODELOS
# ═════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def cargar_cerebros_ia(dataset_id):
    """
    Carga los 3 componentes pre-entrenados desde /pkls:
    - modelo_rf (Random Forest para predicción de RUL)
    - kmeans (para clasificación de regímenes operacionales)
    - features (lista de características esperadas por el modelo)
    """
    try:
        path_mod = os.path.join(FOLDER_PKLS, f'modelo_rf_{dataset_id}.pkl')
        path_km  = os.path.join(FOLDER_PKLS, f'kmeans_{dataset_id}.pkl')
        path_feat = os.path.join(FOLDER_PKLS, f'features_{dataset_id}.pkl')
        
        # Validar que los archivos existan
        if not os.path.exists(path_mod):
            st.error(f"❌ Modelo no encontrado: {path_mod}")
            return None, None, None
        if not os.path.exists(path_km):
            st.error(f"❌ KMeans no encontrado: {path_km}")
            return None, None, None
        if not os.path.exists(path_feat):
            st.error(f"❌ Features no encontrado: {path_feat}")
            return None, None, None
        
        # Cargar los archivos
        modelo = joblib.load(path_mod)
        kmeans = joblib.load(path_km)
        features = joblib.load(path_feat)
        
        st.toast(f"✅ Modelos {dataset_id} cargados correctamente", icon="✨")
        return modelo, kmeans, features
        
    except Exception as e:
        st.error(f"❌ Error al cargar modelos {dataset_id}: {str(e)}")
        return None, None, None


def procesar_y_predecir(df_crudo, modelo, kmeans, features):
    """
    Procesamiento idéntico al Colab:
    1. Renombra columnas de settings
    2. Clasifica regímenes operacionales con KMeans
    3. Normaliza sensores por régimen
    4. Calcula características temporales (media móvil, desv. estándar)
    5. Extrae el último ciclo de cada motor
    6. Realiza predicción del RUL con el modelo Random Forest
    
    Retorna: dict con {id_motor: rul_predicho}
    """
    if modelo is None or kmeans is None or features is None:
        st.error("❌ Los modelos no están cargados. Revisa que los archivos .pkl existan.")
        return {}
    
    df = df_crudo.copy()
    
    # Paso 1: Renombrar columnas de settings para coincidir con entrenamiento
    df.rename(columns={
        'setting_1': 'ajuste1', 
        'setting_2': 'ajuste2', 
        'setting_3': 'ajuste3'
    }, inplace=True)
    
    # Paso 2: Clasificar regímenes operacionales
    try:
        df['regimen'] = kmeans.predict(df[['ajuste1', 'ajuste2', 'ajuste3']])
    except Exception as e:
        st.error(f"❌ Error al predecir regímenes: {e}")
        return {}
    
    # Paso 3 & 4: Normalización y características temporales
    sensores = [c for c in df.columns if 'sensor' in c]
    VENTANA_MOVIL = 15  # Ventana para cálculo de media y desv. móviles
    
    for s in sensores:
        # Normalizar por régimen (min-max scaling dentro de cada régimen)
        df[s] = df.groupby('regimen')[s].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        
        # Media móvil por motor
        df[f'{s}_media_movil'] = df.groupby('id_motor')[s].transform(
            lambda x: x.rolling(VENTANA_MOVIL, min_periods=1).mean()
        )
        
        # Desviación estándar móvil por motor
        df[f'{s}_std_movil'] = df.groupby('id_motor')[s].transform(
            lambda x: x.rolling(VENTANA_MOVIL, min_periods=1).std().fillna(0)
        )
    
    # Paso 5: Tomar último ciclo de cada motor (estado actual)
    df_last = df.groupby('id_motor').last().reset_index()
    
    # Paso 6: Validar que las características requeridas existan
    features_faltantes = [f for f in features if f not in df_last.columns]
    if features_faltantes:
        st.warning(f"⚠️ Características faltantes: {features_faltantes}")
    
    # Preparar matriz X con las features correctas
    X = df_last[features]
    
    # Predicción del RUL
    try:
        predicciones = modelo.predict(X)
        # Asegurar que RUL sea positivo y entero
        predicciones = np.maximum(predicciones, 0)  # RUL >= 0
        rul_dict = dict(zip(df_last['id_motor'].astype(int), np.round(predicciones).astype(int)))
        return rul_dict
    except Exception as e:
        st.error(f"❌ Error durante predicción: {e}")
        return {}


# ═════════════════════════════════════════════════════════════════════════════════
# 3. ESTILOS Y CONFIGURACIÓN VISUAL
# ═════════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #4FA8FF; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stMetricValue"] { color: #4FA8FF; }
    </style>
""", unsafe_allow_html=True)

COLUMNAS_NASA = (
    ['id_motor', 'ciclo', 'setting_1', 'setting_2', 'setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

SENSOR_INFO = {
    'sensor_1':  {'sigla':'T2',        'nombre':'Temperatura entrada Fan',           'unidad':'deg R',   'critico':False},
    'sensor_2':  {'sigla':'T24',       'nombre':'Temperatura salida LPC',            'unidad':'deg R',   'critico':True,  'sube':True},
    'sensor_3':  {'sigla':'T30',       'nombre':'Temperatura salida HPC',            'unidad':'deg R',   'critico':True,  'sube':True},
    'sensor_4':  {'sigla':'T50',       'nombre':'Temperatura salida LPT',            'unidad':'deg R',   'critico':True,  'sube':True},
    'sensor_5':  {'sigla':'P2',        'nombre':'Presion entrada Fan',               'unidad':'psia',    'critico':False},
    'sensor_6':  {'sigla':'P15',       'nombre':'Presion salida Fan',                'unidad':'psia',    'critico':False},
    'sensor_7':  {'sigla':'P30',       'nombre':'Presion salida HPC',                'unidad':'psia',    'critico':True,  'sube':True},
    'sensor_8':  {'sigla':'Nf',        'nombre':'Velocidad fisica del Fan',          'unidad':'rpm',     'critico':True,  'sube':False},
    'sensor_9':  {'sigla':'Nc',        'nombre':'Velocidad fisica del Nucleo',       'unidad':'rpm',     'critico':True,  'sube':True},
    'sensor_10': {'sigla':'epr',       'nombre':'Relacion de presion del motor',     'unidad':'--',      'critico':False},
    'sensor_11': {'sigla':'Ps30',      'nombre':'Presion estatica salida HPC',       'unidad':'psia',    'critico':True,  'sube':True},
    'sensor_12': {'sigla':'phi',       'nombre':'Ratio combustible / presion',       'unidad':'--',      'critico':True,  'sube':True},
    'sensor_13': {'sigla':'NRf',       'nombre':'Velocidad corregida del Fan',       'unidad':'rpm',     'critico':True,  'sube':False},
    'sensor_14': {'sigla':'NRc',       'nombre':'Velocidad corregida del Nucleo',    'unidad':'rpm',     'critico':True,  'sube':True},
    'sensor_15': {'sigla':'BPR',       'nombre':'Relacion de derivacion Bypass',     'unidad':'--',      'critico':True,  'sube':True},
    'sensor_16': {'sigla':'farB',      'nombre':'Ratio aire-combustible quemador',   'unidad':'--',      'critico':False},
    'sensor_17': {'sigla':'htBleed',   'nombre':'Entalpia de purga Bleed',           'unidad':'--',      'critico':True,  'sube':True},
    'sensor_18': {'sigla':'Nf_dmd',    'nombre':'Velocidad demandada del Fan',       'unidad':'rpm',     'critico':False},
    'sensor_19': {'sigla':'PCNfR_dmd', 'nombre':'Vel. demandada corregida Fan',      'unidad':'rpm',     'critico':False},
    'sensor_20': {'sigla':'W31',       'nombre':'Flujo de refrigerante HPT',         'unidad':'lbm/s',   'critico':True,  'sube':True},
    'sensor_21': {'sigla':'W32',       'nombre':'Flujo de refrigerante LPT',         'unidad':'lbm/s',   'critico':True,  'sube':True},
}

sensores_criticos = [k for k, v in SENSOR_INFO.items() if v['critico']]
sensores_estables = [k for k, v in SENSOR_INFO.items() if not v['critico']]

# ═════════════════════════════════════════════════════════════════════════════════
# 4. FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ═════════════════════════════════════════════════════════════════════════════════

def cargar_telemetria(archivo):
    """Carga datos en formato NASA (.txt) o CSV"""
    nombre = archivo.name.lower()
    if nombre.endswith('.txt'):
        df = pd.read_csv(archivo, sep=r'\s+', header=None, engine='python')
        df = df.iloc[:, :26]
        df.columns = COLUMNAS_NASA
    else:
        df = pd.read_csv(archivo)
        if df.columns[0] != 'id_motor':
            df = pd.read_csv(archivo, header=None)
            df = df.iloc[:, :26]
            df.columns = COLUMNAS_NASA
    return df

@st.cache_data
def generar_datos_prueba():
    """Genera datos sintéticos para demostración"""
    filas = []
    for motor in range(1, 6):
        vida_util = np.random.randint(120, 200)
        for ciclo in range(1, vida_util + 1):
            deg = ciclo / vida_util
            s1 = round(np.random.normal(0.0, 0.002), 4)
            s2 = round(np.random.normal(0.0, 0.0003), 4)
            s3 = 100.0
            sensores = [
                round(518.67 + np.random.normal(0, 0.3), 2),
                round(642.0  + deg*3.0  + np.random.normal(0, 0.5), 2),
                round(1587.0 + deg*8.0  + np.random.normal(0, 1.5), 2),
                round(1400.0 + deg*12.0 + np.random.normal(0, 2.0), 2),
                round(14.62  + np.random.normal(0, 0.01), 2),
                round(21.61  + np.random.normal(0, 0.01), 2),
                round(554.0  + deg*4.0  + np.random.normal(0, 0.4), 2),
                round(2388.0 + np.random.normal(0, 0.05), 2),
                round(9050.0 + deg*20.0 + np.random.normal(0, 5.0), 2),
                round(1.30   + np.random.normal(0, 0.003), 2),
                round(47.0   + deg*1.5  + np.random.normal(0, 0.3), 2),
                round(521.0  + deg*2.0  + np.random.normal(0, 0.4), 2),
                round(2388.0 + np.random.normal(0, 0.05), 2),
                round(8130.0 + deg*50.0 + np.random.normal(0, 10.0), 2),
                round(8.42   + deg*0.05 + np.random.normal(0, 0.02), 4),
                round(0.03   + np.random.normal(0, 0.001), 2),
                int(392      + deg*3    + np.random.normal(0, 1)),
                int(2388     + np.random.normal(0, 1)),
                round(100.0  + np.random.normal(0, 0.01), 2),
                round(39.0   + deg*0.5  + np.random.normal(0, 0.2), 2),
                round(23.4   + deg*0.3  + np.random.normal(0, 0.05), 4),
            ]
            filas.append([motor, ciclo, s1, s2, s3] + sensores)
    return pd.DataFrame(filas, columns=COLUMNAS_NASA)

@st.cache_data
def calcular_rul_flota(df):
    """Calcula RUL teórico basado en ciclos máximos"""
    max_c = df.groupby('id_motor')['ciclo'].max().reset_index()
    max_c.columns = ['id_motor', 'max_ciclo']
    df2 = df.merge(max_c, on='id_motor')
    df2['RUL'] = df2['max_ciclo'] - df2['ciclo']
    return df2

@st.cache_data
def calcular_importancia(df_rul):
    """Calcula correlación de sensores con RUL"""
    imp = {}
    for s in sensores_criticos:
        corr = abs(df_rul[s].corr(df_rul['RUL']))
        imp[s] = round(corr, 4) if not np.isnan(corr) else 0.0
    return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

def generar_reporte_csv(df, tabla_rul):
    """Genera reporte en CSV con predicciones"""
    buf = io.StringIO()
    buf.write("REPORTE NASA C-MAPSS - MANTENIMIENTO PREDICTIVO\n\n")
    buf.write("=== PREDICCIONES RUL POR MOTOR ===\n")
    tabla_rul.to_csv(buf, index=False)
    buf.write("\n=== ULTIMOS VALORES POR MOTOR ===\n")
    ultimo = df.sort_values('ciclo').groupby('id_motor').last().reset_index()
    ultimo.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

# ═════════════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR - CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════════

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=150)
st.sidebar.title("Centro de Comando")
st.sidebar.markdown("---")

# Selector de Perfil Operativo (Dataset)
tipo_fd = st.sidebar.selectbox(
    "1. Seleccione Perfil Operativo (Dataset)",
    ("FD001 - Nivel del Mar (HPC)", 
     "FD002 - 6 Condiciones (HPC)",
     "FD003 - Nivel del Mar (HPC + Fan)", 
     "FD004 - 6 Condiciones (HPC + Fan)")
)

# Información sobre el dataset seleccionado
if "FD001" in tipo_fd:
    st.sidebar.info("FD001: Condicion estable (nivel del mar). Desgaste en el HPC.")
elif "FD002" in tipo_fd:
    st.sidebar.info("FD002: 6 condiciones de vuelo. Falla en el HPC.")
elif "FD003" in tipo_fd:
    st.sidebar.info("FD003: Nivel del mar. Fallas en HPC y Fan.")
elif "FD004" in tipo_fd:
    st.sidebar.error("FD004: 6 condiciones + fallas en HPC y Fan.")

st.sidebar.markdown("---")

# Cargador de archivo de telemetría
archivo_subido = st.sidebar.file_uploader(
    "2. Cargar Telemetria (TXT/CSV)", type=['txt', 'csv'],
    help="Formato original NASA (.txt) o CSV con encabezado."
)

st.sidebar.markdown("---")

# Datos de prueba
df_sintetico = generar_datos_prueba()
txt_sintetico = df_sintetico.to_csv(index=False, header=False, sep=' ').encode('utf-8')
st.sidebar.markdown("**Datos de prueba**")
st.sidebar.download_button(
    label="Descargar datos de prueba (.txt)",
    data=txt_sintetico,
    file_name='telemetria_simulada_FD001.txt',
    mime='text/plain',
)

# ═════════════════════════════════════════════════════════════════════════════════
# 6. MAIN - LÓGICA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════════

st.title("Sistema de Mantenimiento Predictivo Aerospacial")
st.markdown("Monitor de degradacion de motores Turbofan - NASA C-MAPSS Dataset.")

# Obtener dataset_id a partir de tipo_fd
dataset_id = FD_MAPPING.get(tipo_fd, "FD001")

# Cargar modelos pre-entrenados
modelo, kmeans, features = cargar_cerebros_ia(dataset_id)

# Validar que tenemos datos a procesar
if archivo_subido is None:
    st.warning("Esperando telemetria. Cargue un archivo .txt (formato NASA) o .csv en el panel izquierdo.")
    st.markdown("#### Asi lucen los datos esperados (muestra sintetica):")
    st.dataframe(df_sintetico.head(10), use_container_width=True)
    st.stop()

# Cargar y procesar telemetría
with st.spinner('Procesando telemetria...'):
    time.sleep(1)
    try:
        df_cargado = cargar_telemetria(archivo_subido)
    except Exception as e:
        st.error("🚨 ¡HOUSTON, TENEMOS UN PROBLEMA! 🚨")
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

if 'id_motor' not in df_cargado.columns:
    st.error("🚨 ¡HOUSTON, TENEMOS UN PROBLEMA! 🚨")
    st.error("El archivo no tiene el formato esperado (columna id_motor no encontrada).")
    st.stop()

# PREDICCIÓN REAL CON EL MODELO
with st.spinner(f'Ejecutando predicciones con modelo {dataset_id}...'):
    rul_predicho_dict = procesar_y_predecir(df_cargado, modelo, kmeans, features)

if not rul_predicho_dict:
    st.error("❌ No se pudieron generar predicciones. Revisa los modelos y datos.")
    st.stop()

df_rul = calcular_rul_flota(df_cargado)
lista_motores = sorted(df_cargado['id_motor'].unique())
st.success(f"✅ Telemetria procesada - {len(df_cargado):,} registros - {len(lista_motores)} motores detectados.")

# Métricas globales
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Motores en Flota", len(lista_motores))
with k2: st.metric("MAE del Modelo", "11.85 ciclos")
with k3: st.metric("Precision R2", "0.84")
with k4:
    cond = "6 Dinamicas" if "6 Condiciones" in tipo_fd else "1 Estatica"
    st.metric("Regimen Operativo", cond)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Diagnostico Individual", "Analisis de Flota", "Base de Datos Cruda", "🚀 Tripulación"])

# ═════════════════════════════════════════════════════════════════════════════════
# TAB 1 - DIAGNÓSTICO INDIVIDUAL
# ═════════════════════════════════════════════════════════════════════════════════
with tab1:
    motor_sel = st.selectbox("Seleccione Motor:", lista_motores, key="motor_tab1")
    datos_motor = (df_cargado[df_cargado['id_motor'] == motor_sel]
                   .copy().sort_values('ciclo').reset_index(drop=True))
    max_ciclo         = int(datos_motor['ciclo'].max())
    min_ciclo         = int(datos_motor['ciclo'].min())
    ciclo_falla       = max_ciclo
    ciclo_degradacion = int(max_ciclo * 0.75)
    VENTANA           = 10

    # Obtener RUL predicho del diccionario del modelo
    rul_predicho = rul_predicho_dict.get(motor_sel, 0)

    col_rul, col_vida, col_crit = st.columns(3)
    with col_rul:
        if rul_predicho > 80:
            st.success(f"### {rul_predicho} ciclos restantes\nESTADO: OPTIMO")
        elif rul_predicho > 30:
            st.warning(f"### {rul_predicho} ciclos restantes\nESTADO: ALERTA")
        else:
            st.error(f"### {rul_predicho} ciclos restantes\nESTADO: CRITICO")
    with col_vida:
        st.metric("Vida util observada", f"{max_ciclo} ciclos")
    with col_crit:
        st.metric("Inicio zona critica", f"ciclo {ciclo_degradacion}")

    st.markdown("<div style='height:4px'></div>", unsafe_html=True)

    ciclo_radar = st.slider(
        "Ciclo para el Radar de Salud",
        min_value=min_ciclo, max_value=max_ciclo,
        value=ciclo_degradacion, step=1,
        key=f"radar_slider_{motor_sel}",
        help="Arrastra para ver la degradacion en cada ciclo"
    )

    pct_vida   = (ciclo_radar - min_ciclo) / max(1, max_ciclo - min_ciclo)
    r_fill     = int(255 * pct_vida)
    g_fill     = int(255 * (1 - pct_vida))
    color_fill = f"rgba({r_fill},{g_fill},0,0.25)"
    color_line = f"rgba({r_fill},{g_fill},0,0.9)"

    fila_ciclo  = datos_motor.iloc[(datos_motor['ciclo'] - ciclo_radar).abs().argsort()[:1]]
    baseline_df = datos_motor.head(10)

    radar_labels, radar_values = [], []
    for s in sensores_criticos:
        info   = SENSOR_INFO[s]
        mu     = baseline_df[s].mean()
        std    = datos_motor[s].std() + 1e-6
        val    = fila_ciclo[s].values[0]
        desv   = (val - mu) / std if info.get('sube', True) else (mu - val) / std
        health = round(max(0.0, min(100.0, 100.0 - desv * 20)), 1)
        radar_labels.append(f"{info['sigla']}<br><sub>{info['nombre'][:22]}</sub>")
        radar_values.append(health)

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself', fillcolor=color_fill,
        mode='lines+markers',
        line=dict(color=color_line, width=2.5),
        marker=dict(size=8, color=color_line, symbol='circle',
                    line=dict(color='white', width=1.2)),
        hovertemplate='<b>%{theta}</b><br>Salud: %{r:.1f}%<extra></extra>',
    ))
    fig_radar.update_layout(
        title=dict(
            text=f"Salud Motor #{motor_sel} - Ciclo {ciclo_radar}/{max_ciclo} ({pct_vida*100:.0f}% vida consumida)",
            font=dict(size=13, color='white'), x=0.5
        ),
        polar=dict(
            bgcolor='rgba(15,20,35,0.8)',
            radialaxis=dict(visible=True, range=[0,100], ticksuffix='%',
                            tickfont=dict(size=9, color='rgba(180,180,180,0.8)'),
                            gridcolor='rgba(120,120,120,0.25)', linecolor='rgba(120,120,120,0.2)'),
            angularaxis=dict(tickfont=dict(size=9, color='white'),
                             gridcolor='rgba(120,120,120,0.2)')
        ),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        margin=dict(t=50, b=30, l=70, r=70), height=430, showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    ultimo = datos_motor[datos_motor['ciclo'] == max_ciclo].iloc[0]
    _e1, col_settings, _e2 = st.columns([2, 3, 2])
    with col_settings:
        st.markdown(
            f"<div style='text-align:center;padding:12px 20px;"
            f"background:rgba(30,40,60,0.75);border-radius:10px;"
            f"border:1px solid rgba(79,168,255,0.25)'>"
            f"<div style='font-size:11px;color:#aaa;margin-bottom:8px;letter-spacing:1px'>"
            f"AJUSTES OPERACIONALES - ULTIMO CICLO</div>"
            f"<table style='width:100%;border-collapse:collapse;font-size:13px'>"
            f"<tr><td style='padding:4px 12px;color:#ccc'>Altitud de vuelo</td>"
            f"    <td style='padding:4px 12px;color:#4FA8FF;font-weight:bold;text-align:right'>"
            f"{ultimo['setting_1']:.4f} ft</td></tr>"
            f"<tr><td style='padding:4px 12px;color:#ccc'>Numero de Mach</td>"
            f"    <td style='padding:4px 12px;color:#4FA8FF;font-weight:bold;text-align:right'>"
            f"{ultimo['setting_2']:.4f} Mach</td></tr>"
            f"<tr><td style='padding:4px 12px;color:#ccc'>TRA (palanca)</td>"
            f"    <td style='padding:4px 12px;color:#4FA8FF;font-weight:bold;text-align:right'>"
            f"{ultimo['setting_3']:.1f} %</td></tr>"
            f"</table></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    with st.expander("Diccionario de Sensores - Referencia rapida", expanded=False):
        filas_dic = []
        for k, v in SENSOR_INFO.items():
            filas_dic.append({
                "N": k.replace("sensor_", "Sensor "),
                "Sigla": v["sigla"],
                "Nombre Real": v["nombre"],
                "Unidad": v["unidad"],
                "Critico": "CRITICO" if v["critico"] else "estable",
            })
        st.dataframe(pd.DataFrame(filas_dic), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Sensores Criticos - Evolucion de Degradacion")

    pares = [sensores_criticos[i:i+2] for i in range(0, len(sensores_criticos), 2)]
    for par in pares:
        cols = st.columns(2)
        for col_obj, s in zip(cols, par):
            info  = SENSOR_INFO[s]
            num_s = s.replace("sensor_", "Sensor ")
            serie = datos_motor.set_index('ciclo')[s]
            suave = serie.rolling(window=VENTANA, center=True, min_periods=1).mean()
            val_f = datos_motor[datos_motor['ciclo'] == max_ciclo][s]
            val_str = f"{val_f.values[0]:.3f} {info['unidad']}" if len(val_f) > 0 else "N/D"

            fig = go.Figure()
            fig.add_vrect(
                x0=ciclo_degradacion, x1=ciclo_falla + 0.5,
                fillcolor="rgba(255,40,40,0.10)", layer="below", line_width=0,
                annotation_text="Zona critica", annotation_position="top left",
                annotation_font=dict(size=8, color="rgba(255,110,110,0.9)")
            )
            fig.add_vline(
                x=ciclo_falla, line_dash="dash",
                line_color="rgba(255,70,70,0.8)", line_width=1.5,
                annotation_text="Falla", annotation_position="top right",
                annotation_font=dict(size=8, color="rgba(255,90,90,0.95)")
            )
            fig.add_trace(go.Scatter(
                x=serie.index, y=serie.values, mode='lines', name='Senal cruda',
                line=dict(color='rgba(100,170,255,0.30)', width=1),
                hovertemplate=f'Ciclo %{{x}}<br>{info["sigla"]}: %{{y:.3f}} {info["unidad"]}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=suave.index, y=suave.values, mode='lines', name='Suavizado',
                line=dict(color='rgba(80,220,255,0.95)', width=2),
                hovertemplate=f'Ciclo %{{x}}<br>{info["sigla"]} suav.: %{{y:.3f}} {info["unidad"]}<extra></extra>'
            ))
            fig.add_annotation(
                xref="paper", yref="paper", x=1.01, y=0.5,
                text=f"<b>{val_str}</b>", showarrow=False, xanchor="left",
                font=dict(size=10, color="rgba(80,220,255,1)")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(18,24,40,0.7)',
                font=dict(color='white', size=10), height=195,
                margin=dict(t=8, b=52, l=45, r=95),
                xaxis=dict(title='Ciclo de Operacion', gridcolor='rgba(80,80,80,0.25)'),
                yaxis=dict(gridcolor='rgba(80,80,80,0.25)'),
                showlegend=True,
                legend=dict(orientation='h', yanchor='top', y=-0.44, xanchor='left', x=0),
            )
            with col_obj:
                st.markdown(
                    f"<p style='margin:0 0 2px 0;font-size:13px;color:#ccc'>"
                    f"<b style='color:#4FA8FF'>{num_s}</b> - "
                    f"<b style='color:white'>{info['sigla']}</b> - {info['nombre']} "
                    f"<span style='color:#888;font-size:11px'>({info['unidad']})</span></p>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Variables Estables")
    st.caption("Sensores sin tendencia de degradacion relevante en este perfil operativo.")
    filas_est = []
    for s in sensores_estables:
        info  = SENSOR_INFO[s]
        num_s = s.replace("sensor_", "Sensor ")
        vals  = datos_motor[s]
        val_f = datos_motor[datos_motor['ciclo'] == max_ciclo][s]
        v_str = f"{val_f.values[0]:.4f}" if len(val_f) > 0 else "N/D"
        filas_est.append({
            "N": num_s, "Sigla": info["sigla"], "Nombre Real": info["nombre"],
            f"Ultimo valor ({info['unidad']})": v_str,
            "Media historica": f"{vals.mean():.4f}",
            "Desv. estandar": f"{vals.std():.5f}",
        })
    st.dataframe(pd.DataFrame(filas_est), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 2 — FLOTA
# ═════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Analisis Global de la Flota")

    tabla_rul = pd.DataFrame([{
        "Motor":             f"Motor {m}",
        "Ciclos observados": int(df_cargado[df_cargado['id_motor'] == m]['ciclo'].max()),
        "RUL Predicho":      rul_predicho_dict.get(m, 0),  # Usar predicciones reales
        "Estado":            ("OPTIMO" if rul_predicho_dict.get(m, 0) > 80 
                            else "ALERTA" if rul_predicho_dict.get(m, 0) > 30 
                            else "CRITICO"),
        "% Vida consumida":  f"{(1 - rul_predicho_dict.get(m, 1)/max(df_cargado[df_cargado['id_motor']==m]['ciclo'].max(),1))*100:.1f}%",
    } for m in lista_motores])

    rul_vals = [rul_predicho_dict.get(m, 0) for m in lista_motores]
    fk1, fk2, fk3, fk4, fk5 = st.columns(5)
    with fk1: st.metric("Total Motores",    len(lista_motores))
    with fk2: st.metric("RUL Promedio",     f"{np.mean(rul_vals):.0f} ciclos")
    with fk3: st.metric("RUL Minimo",       f"{np.min(rul_vals)} ciclos")
    with fk4: st.metric("Motores Criticos", sum(1 for r in rul_vals if r <= 30))
    with fk5: st.metric("En Alerta",        sum(1 for r in rul_vals if 30 < r <= 80))

    st.markdown("---")

    # Barras RUL
    st.markdown("#### RUL Predicho por Motor")
    colores_rul = ['#e74c3c' if r <= 30 else '#f39c12' if r <= 80 else '#2ecc71' for r in rul_vals]
    fig_bar = go.Figure(go.Bar(
        x=[f"Motor {m}" for m in lista_motores], y=rul_vals,
        marker_color=colores_rul, text=rul_vals, textposition='outside',
        hovertemplate='<b>%{x}</b><br>RUL: %{y} ciclos<extra></extra>'
    ))
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(18,24,40,0.7)',
        font=dict(color='white', size=11), height=300,
        margin=dict(t=20, b=40, l=50, r=20),
        xaxis=dict(gridcolor='rgba(80,80,80,0.2)'),
        yaxis=dict(title='Ciclos restantes', gridcolor='rgba(80,80,80,0.2)'),
        showlegend=False,
    )
    fig_bar.add_hline(y=30, line_dash='dash', line_color='red',
                      annotation_text='Umbral critico (30)',
                      annotation_font=dict(color='red', size=10))
    fig_bar.add_hline(y=80, line_dash='dash', line_color='orange',
                      annotation_text='Umbral alerta (80)',
                      annotation_font=dict(color='orange', size=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Sensores toda la flota
    st.markdown("#### Comportamiento de Sensores Criticos - Toda la Flota")
    sensores_labels = {
        s: f"{s.replace('sensor_','Sensor ')} - {SENSOR_INFO[s]['sigla']} - {SENSOR_INFO[s]['nombre']}"
        for s in sensores_criticos
    }
    sensor_flota = st.selectbox(
        "Sensor a comparar entre motores:",
        list(sensores_labels.keys()),
        format_func=lambda x: sensores_labels[x],
        index=10, key="sensor_flota"
    )
    info_sf = SENSOR_INFO[sensor_flota]
    paleta  = px.colors.qualitative.Plotly

    fig_flota = go.Figure()
    for i, m in enumerate(lista_motores):
        dm    = df_cargado[df_cargado['id_motor'] == m].sort_values('ciclo')
        serie = dm.set_index('ciclo')[sensor_flota]
        suave = serie.rolling(window=10, center=True, min_periods=1).mean()
        color = paleta[i % len(paleta)]
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig_flota.add_trace(go.Scatter(
            x=serie.index, y=serie.values, mode='lines', name=f'M{m} crudo',
            line=dict(color=f'rgba({r},{g},{b},0.20)', width=1), showlegend=False,
        ))
        fig_flota.add_trace(go.Scatter(
            x=suave.index, y=suave.values, mode='lines', name=f'Motor {m}',
            line=dict(color=color, width=2),
            hovertemplate=f'Motor {m} Ciclo %{{x}}<br>{info_sf["sigla"]}: %{{y:.3f}}<extra></extra>'
        ))
    fig_flota.update_layout(
        title=dict(text=f"{info_sf['sigla']} - {info_sf['nombre']} ({info_sf['unidad']})",
                   font=dict(size=13, color='white'), x=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(18,24,40,0.7)',
        font=dict(color='white', size=10), height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        xaxis=dict(title='Ciclo de Operacion', gridcolor='rgba(80,80,80,0.25)'),
        yaxis=dict(title=info_sf['unidad'], gridcolor='rgba(80,80,80,0.25)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    st.plotly_chart(fig_flota, use_container_width=True)

    st.markdown("---")

    # Importancia de sensores
    st.markdown("#### Impacto de Sensores en la Vida Util (Importancia de Variables)")
    importancias = calcular_importancia(df_rul)
    imp_labels, imp_values, imp_colors = [], [], []
    for s, val in importancias.items():
        info = SENSOR_INFO[s]
        imp_labels.append(f"{s.replace('sensor_','S')} - {info['sigla']} - {info['nombre'][:30]}")
        imp_values.append(val)
        imp_colors.append(
            '#e74c3c' if val >= 0.7 else
            '#e67e22' if val >= 0.5 else
            '#f1c40f' if val >= 0.3 else '#3498db'
        )

    fig_imp = go.Figure(go.Bar(
        x=imp_values, y=imp_labels, orientation='h',
        marker=dict(color=imp_colors),
        text=[f"{v:.3f}" for v in imp_values], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Correlacion con RUL: %{x:.4f}<extra></extra>'
    ))
    fig_imp.update_layout(
        title=dict(text="Correlacion Absoluta con el RUL (proxy de importancia SHAP)",
                   font=dict(size=13, color='white'), x=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(18,24,40,0.7)',
        font=dict(color='white', size=10), height=520,
        margin=dict(t=45, b=40, l=310, r=80),
        xaxis=dict(title='|Correlacion con RUL|', gridcolor='rgba(80,80,80,0.25)',
                   range=[0, max(imp_values) * 1.20]),
        yaxis=dict(gridcolor='rgba(80,80,80,0.15)', autorange='reversed'),
        showlegend=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        "<div style='background:rgba(30,40,60,0.75);border-radius:10px;"
        "border-left:4px solid #4FA8FF;padding:16px 20px;margin-top:-10px'>"
        "<b style='color:#4FA8FF;font-size:14px'>Que es la Importancia de Variables (SHAP)?</b><br><br>"
        "<span style='color:#ccc;font-size:13px'>"
        "<b>SHAP</b> (SHapley Additive exPlanations) cuantifica cuanto contribuye cada sensor "
        "a la prediccion del modelo de IA. En este grafico se muestra la "
        "<b>correlacion absoluta</b> de cada sensor con el <b>RUL</b>: "
        "cuanto mayor la barra, mas cambia ese sensor al acercarse la falla.<br><br>"
        "<b style='color:#e74c3c'>Rojo</b>: impacto muy alto (>=0.7) | "
        "<b style='color:#e67e22'>Naranja</b>: alto (>=0.5) | "
        "<b style='color:#f1c40f'>Amarillo</b>: moderado (>=0.3) | "
        "<b style='color:#3498db'>Azul</b>: bajo"
        "</span></div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("#### Tabla de Predicciones RUL por Motor")
    st.dataframe(tabla_rul, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    reporte_bytes = generar_reporte_csv(df_cargado, tabla_rul)
    st.download_button(
        label="Descargar Informe Completo (.csv)",
        data=reporte_bytes,
        file_name='informe_mantenimiento_predictivo.csv',
        mime='text/csv',
    )


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATOS CRUDOS
# ═════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📂 Explorador de Telemetría (Datos Crudos)")
    
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    
    if col_btn1.button("✅ Sel. Todos"):
        st.session_state.motores_filtro = list(lista_motores)
    
    if col_btn2.button("🗑️ Limpiar"):
        st.session_state.motores_filtro = []

    if 'motores_filtro' not in st.session_state:
        st.session_state.motores_filtro = [lista_motores[0]]

    motores_seleccionados = st.multiselect(
        "Filtrar por ID de Motor:",
        options=lista_motores,
        key="motores_filtro",
        help="Selecciona los motores para ver su telemetría completa."
    )

    if motores_seleccionados:
        df_filtrado = df_cargado[df_cargado['id_motor'].isin(motores_seleccionados)]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas en vista", f"{len(df_filtrado)}")
        c2.metric("Motores", f"{len(motores_seleccionados)}")
        c3.info("💡 Tip: Haz clic en las columnas para ordenar.")

        st.dataframe(
            df_filtrado, 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("💡 Usa los botones de arriba o selecciona un motor para inspeccionar los datos.")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRIPULACIÓN
# ═════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_icon, col_title = st.columns([1, 8])
    with col_icon:
        st.image("https://cdn-icons-png.flaticon.com/512/6741/6741064.png", width=80)
    with col_title:
        st.title("Tripulación de la Misión")
        st.subheader("Aprendizaje Automático I")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👨‍🚀 Antonio Narváez")
        st.caption("Administrador de Empresas")
        st.write("""
        **Especialista en Ciencia de Datos** Líder del proyecto y arquitecto de la lógica de negocio aplicada al sector salud y mantenimiento.
        """)
    
    with col2:
        st.markdown("### 👩‍🚀 Jocelyn Rodríguez")
        st.caption("Ingeniera Industrial")
        st.write("""
        **Optimización de Procesos** Experta en modelamiento de flujos operativos y gestión de restricciones industriales.
        """)

    st.markdown("---")

    c_info1, c_info2 = st.columns(2)

    with c_info1:
        st.markdown("#### 🤖 Aumentado por IA")
        st.info("""
        Desarrollado utilizando prácticas modernas de ingeniería asistida por IA para garantizar 
        una interfaz (UI/UX) óptima y una implementación matemática limpia.
        """)
        
        st.markdown("#### 🔒 Privacidad y Procesamiento Local")
        st.info("""
        Esta herramienta está intencionalmente desacoplada de servicios en la nube externos. 
        Todos los datos de producción y restricciones se calculan directamente en su dispositivo.
        """)

    with c_info2:
        st.markdown("#### ⚙️ Lógica de Producción")
        st.info("""
        Enfocado en los fundamentos del machine learning, proporcionando una visualización 
        del dataset utilizado por un simulador para evaluar el desgaste en sus motores.
        """)
        
        st.markdown("#### 🎯 Propósito Educativo")
        st.success("""
        Desarrollar un sistema basado en Machine Learning capaz de estimar la vida útil restante (RUL) de motores aeronáuticos a partir de datos de sensores, e implementar un Producto Mínimo Viable (PMV) que permita visualizar el estado de degradación de los motores mediante un dashboard interactivo.
        """)

    st.markdown("<center><small>NASA C-MAPSS Predictor v3.0 | 2026</small></center>", unsafe_allow_html=True)
