import streamlit as st
import pandas as pd  
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns
import rasterio
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from xgboost import XGBRegressor, XGBClassifier
import requests
import pydeck as pdk
import os
import sys
import base64

#import appk001ubicacion
#import appk002zonaestud
#import appk003preproces
#import appk004modelalit
#import appk005pronoslit 
#import appk006modlog10k
#import appk007prolog10k
#import appk008graficars

# ==========================================================================================================================================
# CONFIGURACIÓN DE LA PÁGINA (Única declaración global)
# ==========================================================================================================================================
st.set_page_config(page_title="Gemelos Digitales - Acuíferos", layout="wide")

mapbox_key = st.secrets.get("MAPBOX_TOKEN", "")

# ==========================================================================================================================================
# MECANISMO DE CACHÉ Y RUTAS: OPTIMIZACIÓN DE IMÁGENES Y LOGOS EN RAM
# ==========================================================================================================================================
# Detectamos la ruta absoluta del directorio actual en el servidor
ruta_base = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

@st.cache_data(show_spinner=False)
def obtener_logo_html_central(ruta_img):
    if os.path.exists(ruta_img):
        try:
            with open(ruta_img, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            return f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 60vh; flex-direction: column;">
                <a href="?ingresar=true" target="_self">
                    <img src="data:image/png;base64,{img_base64}" 
                         style="width: 500px; max-width: 100%; height: auto; cursor: pointer; transition: transform 0.3s ease;"
                         onmouseover="this.style.transform='scale(1.03)'" 
                         onmouseout="this.style.transform='scale(1)'">
                </a>
            </div>
            """
        except Exception:
            pass
            
    return """
    <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 60vh; flex-direction: column;">
        <a href="?ingresar=true" target="_self" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #1E3A8A, #3B82F6); color: white; padding: 25px 60px; 
                        font-size: 28px; font-weight: bold; border-radius: 50px; cursor: pointer; 
                        box-shadow: 0 10px 20px rgba(59,130,246,0.3); text-align: center; transition: 0.3s;">
                🔵 ENTER TO NEREUS <br>
                <span style="font-size: 14px; font-weight: normal; opacity: 0.8;">(Click to start application)</span>
            </div>
        </a>
    </div>
    """

@st.cache_data(show_spinner=False)
def cargar_logo_fijo(ruta_archivo):
    try:
        return Image.open(ruta_archivo)
    except FileNotFoundError:
        return None

# --- CARGA INTELIGENTE DE LOGOS EN MEMORIA ---

# 1. Logo Central de Bienvenida (Usa la ruta absoluta calculada)
ruta_logo_nereus = os.path.join(ruta_base, "logo_nereus.png")

# 2. Logo Licenciado del cuerpo principal (Mantenemos PIL)
logo_nuevo_cache = cargar_logo_fijo(os.path.join(ruta_base, "logo_licenciado.png")) or cargar_logo_fijo(os.path.join(ruta_base, "logo_nuevo.jpg"))

# 3. 🌟 LOGO LATERAL CORREGIDO: Carga directa por ruta del sistema para máxima compatibilidad
ruta_logo_sidebar = os.path.join(ruta_base, "logo_nereus2.png")
logo_nereus2_cache = ruta_logo_sidebar if os.path.exists(ruta_logo_sidebar) else None


# ==========================================================================================================================================
# GESTIÓN DE MEMORIA RAM (Session State)
# ==========================================================================================================================================
if "app_iniciada" not in st.session_state:
    st.session_state["app_iniciada"] = False
if "K001_datos" not in st.session_state:
    st.session_state["K001_datos"] = None
if "K001_dem" not in st.session_state:
    st.session_state["K001_dem"] = None
if "procesar_click" not in st.session_state:
    st.session_state["procesar_click"] = False

# ==========================================================================================================================================
# PANTALLA DE BIENVENIDA (LOGO NEREUS COMO BOTÓN ACCIONABLE)
# ==========================================================================================================================================
query_params = st.query_params
if "ingresar" in query_params:
    st.session_state["app_iniciada"] = True
    st.query_params.clear()
    st.rerun()

if not st.session_state["app_iniciada"]:
    st.markdown("""
        <style>
        .block-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        </style> 
    """, unsafe_allow_html=True)

    html_logo_boton = obtener_logo_html_central(ruta_logo_nereus)
    st.markdown(html_logo_boton, unsafe_allow_html=True)
   
    st.stop()  # <--- AQUÍ SE DETIENE LA BIENVENIDA. NADIE PASA DE AQUÍ SIN DAR CLICK.


# ==========================================================================================================================================
# 🌟 TOPE ABSOLUTO DEL CUERPO (SE EJECUTA ÚNICAMENTE DESPUÉS DE LA BIENVENIDA)
# ==========================================================================================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_logo_1, col_logo_2, col_logo_3 = st.columns([1, .5, 1])
with col_logo_2:
    if logo_nuevo_cache is not None:
        st.image(logo_nuevo_cache, use_container_width=True)
    else:
        st.markdown(
            "<div style='text-align:center; padding:15px; background:#f0f2f6; border-radius:10px; color:#333; font-weight:bold; margin-bottom:15px;'> "
            "✨ [ LOGO LICENCIADO ] ✨"
            "</div>", 
            unsafe_allow_html=True
        )


# ==========================================================================================================================================
# 1. BARRA LATERAL (SIDEBAR) - DISPONIBLE SÓLO SI LA APP YA FUE INICIADA
# ==========================================================================================================================================
with st.sidebar:
    st.markdown("""

    <style>
    /* Fondo General Sidebar */
    [data-testid="stSidebar"] { background-color: #F4F6F9 !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h3 { color: #1E293B !important; }
    
    /* Contenedor Formulario (Azul Marino Oscuro) */
    [data-testid="stSidebar"] [data-testid="stForm"] {
        background-color: #0F172A !important; 
        border: 1px solid #334155; border-radius: 12px; padding: 20px;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] h3 { color: #FFFFFF !important; }
    [data-testid="stSidebar"] [data-testid="stForm"] label { color: #E2E8F0  !important; } ##E2E8F0 

    /* Zonas de Carga (Azul Acero Claro con contraste alto) */
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stFileUploaderDropzone"] {
        background-color: #0EA5E9 !important;  #1E293B
        border: 2px dashed #38BDF8 !important; border-radius: 8px !important; #solid
    }
    /* Texto Informativo de Archivos (Blanco Nítido) */
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stFileUploaderDropzone"] p {
        color: #F1F5F9 !important; font-size: 13px !important;
    }

    /* Botón interno Upload */
    [data-testid="stSidebar"] [data-testid="stForm"] .stFileUploader button {
        background-color: #0F172A !important; color: #38BDF8 !important; border: 1px solid #38BDF8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] .stFileUploader button svg { fill: #38BDF8 !important; }

    /* Botón Principal Procesar Datos (Celeste Tecnológico) */
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stBaseButton-secondaryFormSubmit"] {
        background-color: #0EA5E9 !important; color: #FFFFFF !important; font-weight: bold !important; border: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        background-color: #0284C7 !important;
    }
    </style>
""", unsafe_allow_html=True)






    
    if logo_nereus2_cache is not None:
        st.image(logo_nereus2_cache, width=180)  
        #st.markdown("---")
    else:
        st.warning("⚠️ Logo 'logo_nereus2.png' no disponible en el servidor.")
        st.markdown("---")
        
    #st.write("### Panel de Datos de Entrada")
    
    if st.button("⬅️ Cerrar Sesión / Inicio", use_container_width=True):
        st.session_state["app_iniciada"] = False
        st.session_state["procesar_click"] = False
        st.rerun()
        
    with st.form("formulario_carga"):
        st.write("### Panel de datos de entrada")
        archivo_csv = st.file_uploader("Arrastra y suelta o click para subir archivos CSV", type=["csv"])
        archivo_tif = st.file_uploader("Arrastra y suelta o click para subir archivos TIF", type=["tif", "tiff"])
        #st.markdown("---")
        boton_procesar = st.form_submit_button("Procesar datos")

# ==========================================================================================================================================
# 2. GESTIÓN DE ENTRADA DE DATOS
# ==========================================================================================================================================
if boton_procesar:
    st.session_state["procesar_click"] = True
    if archivo_csv is not None:
        try:
            st.session_state["K001_datos"] = pd.read_csv(archivo_csv)
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV: {e}")
    else:
        st.session_state["K001_datos"] = None
        
    if archivo_tif is not None:
        try:
            st.session_state["K001_dem"] = archivo_tif.read()
        except Exception as e:
            st.sidebar.error(f"Error al leer TIF: {e}")
    else:
        st.session_state["K001_dem"] = None

# Estado de validación
ambos_archivos_listos = st.session_state["K001_datos"] is not None and st.session_state["K001_dem"] is not None

if not ambos_archivos_listos:
    st.session_state["procesar_click"] = False

# Extracción segura desde el Session State
K001_dem = st.session_state["K001_dem"]
K001_datos = st.session_state["K001_datos"]


# ==========================================================================================================================================
# 3. CONTENIDO CENTRAL DE LA PÁGINA (ESTADO ACTIVO - ABAJO DEL LOGO LICENCIADO)
# ==========================================================================================================================================
# Control de flujo de datos por debajo de la cabecera unificada
if not ambos_archivos_listos:
    st.markdown("<h4 style='text-align: center; color: #555;'>💡 Por favor, en la barra lateral izquierda cargar los archivos CSV y TIF y presionar 'PROCESAR DATOS'.</h4>", unsafe_allow_html=True)
else:
    if not st.session_state["procesar_click"]:
        st.write("")
    else:
        st.markdown("<h3 style='text-align: center; color: #1E3A8A; font-size: 32px; font-weight: bold;'>Gemelos Digitales - Gestión de Acuíferos</h3>", unsafe_allow_html=True)

        # Renderizado de los Múltiples Tabs de la Plataforma
        tabs = st.tabs([
            "Ubicación", "Zona de Pronóstico", "Preprocesamiento", 
            "Modelamiento Litológico", "Pronóstico Litológico",  
            "Modelamiento de Log10 K", "Pronóstico Log10 K",  
            "Gráfico 3D"
        ])
        
        with tabs[0]: # Ubicación
            appk001ubicacion.BLOQUE001(K001_datos, K001_dem)
            appk001ubicacion.BLOQUE002(K001_datos, mapbox_key) 
            appk001ubicacion.BLOQUE003(K001_datos, K001_dem, submuestreo=5)
            
        with tabs[1]: # Zona de Pronóstico
            appk002zonaestud.BLOQUE001(K001_dem) 
            
        with tabs[2]: # Preprocesamiento
            appk003preproces.BLOQUE001() 
            appk003preproces.BLOQUE002() 
            appk003preproces.BLOQUE003() 
            
        with tabs[3]: # Modelamiento Litológico
            appk004modelalit.BLOQUE001() 
            
        with tabs[4]: # Pronóstico Litológico  
            appk005pronoslit.BLOQUE001() 
            
        with tabs[5]: # Modelamiento de Log10 K    
            #appk006modlog10k.BLOQUE001() 
            appk006modlog10k.BLOQUE002()
            
        with tabs[6]: # Pronóstico Log10 K    
            appk007prolog10k.BLOQUE001() 
            
        with tabs[7]: # Gráfico 3D    
            appk008graficars.BLOQUE001()
            appk008graficars.BLOQUE002()


