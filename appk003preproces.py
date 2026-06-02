import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib


#***********************************************************************************************************************************************
#***********************************************************************************************************************************************
def BLOQUE001():
    st.markdown("<h4 style='text-align: center;'>Limpieza de Data</h4>", unsafe_allow_html=True)
    st.markdown("Transformación automática de **K001_datos** a **K002_datos_limpio**.")

    # 1. VERIFICAR SI EXISTEN LOS DATOS EN LA SESIÓN
    if 'K001_datos' not in st.session_state or st.session_state['K001_datos'] is None:
        st.warning("⚠️ No se encontraron datos cargados en 'K001_datos'.")
        st.info("Por favor, sube primero el archivo CSV requerido en la barra lateral.")
        return

    # 2. RECUPERAR DATOS DE LA SESIÓN
    df_raw = st.session_state['K001_datos'].copy()

    # --- VALIDACIÓN DEFENSIVA DE COLUMNAS ---
    # Limpieza de espacios en los nombres de las columnas
    df_raw.columns = df_raw.columns.str.strip()
    
    # Lista de columnas requeridas estrictamente en esta fase
    cols_ubicacion = ['Longitud', 'Latitud', 'Altitud']
    cols_agregacion = ['K', 'Tipo_Roca', 'Profundidad', 'Cota']
    columnas_requeridas = cols_ubicacion + cols_agregacion

    # Verificar cuáles columnas faltan en el DataFrame real
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_raw.columns]

    if columnas_faltantes:
        st.error("❌ **Error en la estructura del archivo CSV**")
        st.markdown(f"El archivo cargado no contiene todas las columnas necesarias para la limpieza.")
        st.warning(f"**Columnas faltantes:** `{columnas_faltantes}`")
        
        with st.expander("💡 Columnas detectadas en tu archivo actual"):
            st.write(list(df_raw.columns))
            st.info("Asegúrate de que los encabezados de tu CSV coincidan exactamente con: "
                    "`Longitud`, `Latitud`, `Altitud`, `K`, `Tipo_Roca`, `Profundidad`, `Cota`")
        return 

    # --- PROCESAMIENTO ---
    # Agrupamos para promediar K en el mismo punto 3D sin HGS ni Es_Zona_Compleja
    df = df_raw.groupby(cols_ubicacion, as_index=False).agg({
        'K': 'mean',
        'Tipo_Roca': 'first',
        'Profundidad': 'mean',
        'Cota': 'first'
    })

    # Manejo de ceros y Transformación Logarítmica
    df['K'] = df['K'].replace(0, 1e-10)
    df['log10_K'] = np.log10(df['K'])

    # Tratamiento de Outliers (IQR)
    Q1 = df['log10_K'].quantile(0.25)
    Q3 = df['log10_K'].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    # Aplicar Capping (Limitar valores extremos)
    df['log10_K'] = df['log10_K'].clip(lower=lim_inf, upper=lim_sup)
    
    # Estandarización de texto
    df['Tipo_Roca'] = df['Tipo_Roca'].astype(str).str.strip().str.capitalize()

    # Selección final de columnas corregida
    df_final = df[['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota', 'Tipo_Roca', 'log10_K']]

    # --- INTERFAZ DE RESULTADOS ---
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros Originales", len(df_raw))
        c2.metric("Puntos Únicos (Final)", len(df_final))
        c3.metric("Límite Inf (log10)", f"{lim_inf:.2f}")
        c4.metric("Límite Sup (log10)", f"{lim_sup:.2f}")

    # --- VISUALIZACIÓN ---
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>📊 Distribución Normalizada</h4>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_final, 
            x="log10_K", 
            color="Tipo_Roca", 
            marginal=None, 
            nbins=30, 
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={'log10_K': 'Permeabilidad (log10_K)', 'count': 'Frecuencia'}
        )

        fig_hist.update_layout(
            bargap=0.1,                
            height=500,                
            margin=dict(t=30, b=30),
            legend_title_text='Tipo de Roca',
            yaxis=dict(domain=[0, 1])
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Calcular la tabla de frecuencias basándonos en la data limpia generada
        df_frecuencia = df_final['Tipo_Roca'].value_counts().reset_index()
        df_frecuencia.columns = ['Tipo_Roca', 'Frecuencia_Absoluta']
    
        # Agregar porcentaje
        total = df_frecuencia['Frecuencia_Absoluta'].sum()
        df_frecuencia['Porcentaje (%)'] = ((df_frecuencia['Frecuencia_Absoluta'] / total) * 100).round(2)

        st.markdown("<h4 style='text-align: center;'>📊 Resumen de Litologías</h4>", unsafe_allow_html=True)
        st.dataframe(df_frecuencia, use_container_width=True)

        csv_frecuencia = df_frecuencia.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Descargar Tabla de Frecuencias en CSV",
            data=csv_frecuencia,
            file_name="Frecuencia_Tipo_Roca_K002.csv",
            mime="text/csv",
            use_container_width=True
        )

        html_bytes_hist = fig_hist.to_html(include_plotlyjs='cdn').encode('utf-8')
    
        st.download_button(
            label="📥 Descargar Histograma Interactivo HTML",
            data=html_bytes_hist,
            file_name="Distribucion_Normalizada_K.html",
            mime="text/html",
            use_container_width=True
        )
    
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>📈 K por Litología</h4>", unsafe_allow_html=True)
        fig_box = px.box(df_final, x="Tipo_Roca", y="log10_K", color="Tipo_Roca",
                         template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
        
        st.plotly_chart(fig_box, use_container_width=True)

        html_bytes_box = fig_box.to_html().encode('utf-8')

        st.download_button(
            label="📥 Descargar Gráfico Interactivo HTML",
            data=html_bytes_box,
            file_name="K_por_Litologia.html",
            mime="text/html",
            use_container_width=True
        )

    # --- TABLA Y DESCARGA FINAL ---
    with st.expander("🔍 Ver tabla de datos"):
        st.dataframe(df_final, use_container_width=True)

    col_btn, col_label = st.columns([1, 1])
    csv_data = df_final.to_csv(index=False).encode('utf-8')

    with col_btn:
        st.download_button(
            label="📥 Descargar K002_datos_limpio.csv",
            data=csv_data,
            file_name="K002_datos_limpio.csv",
            mime="text/csv",
            use_container_width=True
        )
 
    # --- GUARDAR EN SESIÓN PARA EL SIGUIENTE PASO ---
    st.session_state['K002_datos_limpio'] = df_final.copy()
    st.toast("Datos normalizados y almacenados en K002_datos_limpio", icon="🧼")


#***********************************************************************************************************************************************
#***********************************************************************************************************************************************
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def BLOQUE002(): #INGENIERIA DE CARACTERISTICAS
    st.markdown("<h4 style='text-align: center;'>Ingeniería de Rocas por Localización (HGS)</h4>", unsafe_allow_html=True)

    # 1. VERIFICAMOS QUE LA DATA LIMPIA DE ENTRADA EXISTA EN SESIÓN (Entrada: K002_datos_limpio)
    if 'K002_datos_limpio' not in st.session_state or st.session_state['K002_datos_limpio'] is None:
        st.warning("⚠️ No se encontraron los datos limpios en 'K002_datos_limpio'.")
        st.info("Por favor, completa primero el procesamiento en la pestaña de **Limpieza de Data (K002)**.")
        return

    # Recuperamos los datos de entrada limpios (escala real)
    df_raw = st.session_state['K002_datos_limpio']

    with st.container(border=True):
        # --- 1. PARAMETRIZACIÓN (Sliders en la barra lateral) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Ajustes de DBSCAN (HGS)")
        
        # Uso de keys únicas para evitar conflictos en el estado de Streamlit
        val_eps = st.sidebar.slider("Radio de vecindad (eps)", 0.1, 5.0, 0.5, step=0.1, key="dbscan_eps")
        val_min_samples = st.sidebar.slider("Muestras mínimas (min_samples)", 1, 20, 5, key="dbscan_min")

        # --- 2. PROCESAMIENTO ---
        df = df_raw.copy() 
        features_cols = ['Profundidad', 'Longitud', 'Latitud', 'Altitud']
 
        if all(col in df.columns for col in features_cols):
            # 💡 IMPORTANTE: Como los datos vienen limpios pero en escala real (K002), 
            # aplicamos un escalado temporal rápido solo para calcular las distancias de DBSCAN correctamente.
            scaler_temporal = StandardScaler()
            features_scaled = scaler_temporal.fit_transform(df[features_cols])
    
            # Ejecución de Clustering espacial basado en densidad (Generación de unidades Litológicas/Roca)
            dbscan = DBSCAN(eps=val_eps, min_samples=val_min_samples)
            df['HGS'] = dbscan.fit_predict(features_scaled)
        
            # Identificación de ruido (Mapeado a código 999 para Zonas Complejas)
            df['HGS'] = df['HGS'].replace(-1, 999)
            df['Es_Zona_Compleja'] = (df['HGS'] == 999).astype(int)

            # --- 3. MÉTRICAS DE INTERFAZ ---
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Registros", len(df))
                m2.metric("Unidades de Roca (HGS)", len(df[df['HGS'] != 999]['HGS'].unique()))
                m3.metric("Zonas Complejas (Ruido)", int(df['Es_Zona_Compleja'].sum()), delta_color="inverse")

            with st.container(border=True):
                # --- 4. VISUALIZACIÓN 3D (Se grafica usando las coordenadas reales para coherencia visual) ---
                st.markdown("<h4 style='text-align: center;'>Visualización Espacial de Estructuras Rocosas</h4>", unsafe_allow_html=True)
          
                df['Etiqueta'] = df['HGS'].apply(lambda x: "Zona Compleja" if x == 999 else f"Tipo Roca {x}")
    
                fig = px.scatter_3d(
                    df, 
                    x='Longitud', 
                    y='Latitud', 
                    z='Altitud',
                    color='Etiqueta',
                    symbol='Es_Zona_Compleja',
                    title=f"Estratificación HGS (eps={val_eps}, min_samples={val_min_samples})",
                    labels={'Etiqueta': 'Clasificación Geológica'},
                    opacity=0.7,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
    
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=30), 
                    scene_dragmode='orbit',
                    height=650
                )
            
                st.plotly_chart(fig, use_container_width=True)

            # --- 5. TABLA Y DESCARGA ---
            with st.expander("🔍 Ver vista previa de las unidades de roca asignadas (Primeros 20 registros)"):
                st.dataframe(df.head(20), use_container_width=True)

            col_html, col_csv = st.columns(2)

            with col_html:
                html_bytes = fig.to_html().encode('utf-8')
                st.download_button(
                    label="📥 Descargar Mapa Interactivo HTML",
                    data=html_bytes,
                    file_name="NEREUS_Mapa_3D_HGS.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col_csv:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar K003_hgs.csv",
                    data=csv_data,
                    file_name='K003_hgs.csv',  # Adaptado al requerimiento
                    mime='text/csv',
                    use_container_width=True
                )

            # --- 6. ALMACENAMIENTO DE SALIDA (Guardado en memoria con la nueva nomenclatura) ---
            st.session_state['K003_hgs'] = df.copy()
            st.caption("✅ Ingeniería de características completada. Datos guardados en memoria como **K003_hgs**.")
                                                 
        else:
            st.error(f"⚠️ Estructura inválida. Faltan columnas espaciales en el dataset de entrada. Se requiere: {features_cols}")

#***********************************************************************************************************************************************
#***********************************************************************************************************************************************
import streamlit as st
import pandas as pd
import io
import joblib
from sklearn.preprocessing import StandardScaler

def BLOQUE003(): # Reescalado (Actualizado a Bloque K004)
    st.markdown("<h4 style='text-align: center;'>Reescalado de Variables Numéricas</h4>", unsafe_allow_html=True)

    st.info("Se aplica una estandarización (Z-score) únicamente a las variables físicas continuas para normalizar su media a 0 y su desviación estándar a 1. Las variables de ingeniería HGS se mantienen intactas.")

    # 1. VERIFICAR DATOS DE LA SESIÓN (Entrada: K003_hgs)
    if 'K003_hgs' not in st.session_state or st.session_state['K003_hgs'] is None:
        st.warning("⚠️ No se encontraron los datos con ingeniería HGS en 'K003_hgs'.")
        st.info("Por favor, completa primero el procesamiento en la pestaña de **Ingeniería de Rocas (K003)**.")
        return

    # Hacemos una copia profunda del dataframe de entrada
    df = st.session_state['K003_hgs'].copy()
    
    # 🌟 CORRECCIÓN: Definimos estrictamente las columnas que pasarán por el StandardScaler
    cols_a_escalar = ['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota']

    # 2. PROCESAMIENTO: ESCALADO
    scaler_ml = StandardScaler()
    
    try:
        # Validar que todas las columnas espaciales requeridas existan antes de transformar
        missing_cols = [col for col in cols_a_escalar if col not in df.columns]
        if missing_cols:
            raise KeyError(f"{missing_cols}")

        # Se aplica la transformación solo a las 5 variables físicas
        df[cols_a_escalar] = scaler_ml.fit_transform(df[cols_a_escalar])
        
        # Guardamos en Session State utilizando la nomenclatura correlativa (Salidas K004)
        st.session_state['K004_escalear'] = df
        st.session_state['K004_scaler_ml'] = scaler_ml

        st.success("✅ Coordenadas y variables físicas reescaladas correctamente. Guardado internamente como K004_escalear.")

    except KeyError as e:
        st.error(f"❌ Error: No se encontraron las columnas numéricas {e} en el dataset con ingeniería HGS.")
        return

    # 3. INTERFAZ DE USUARIO: ESTADÍSTICAS
    with st.container(border=True):
        st.write("**Estadísticas post-escalado (Media ≈ 0, Desv. Est. ≈ 1):**")
        st.dataframe(df[cols_a_escalar].describe().loc[['mean', 'std']], use_container_width=True)

        # 4. PREPARAR DESCARGAS PARA USO FUTURO
        col1, col2 = st.columns(2)

        # Descarga CSV homologado a K004_escalear.csv
        csv_esc = df.to_csv(index=False).encode('utf-8')
        col1.download_button(
            label="📥 Descargar K004_escalear.csv",
            data=csv_esc,
            file_name="K004_escalear.csv",
            mime="text/csv",
            use_container_width=True,
            help="Descarga el dataset con las variables físicas normalizadas y las etiquetas HGS en formato original."
        )

        # Descarga del Scaler (Joblib) homologado a K004_scaler_ml.joblib
        scaler_buffer = io.BytesIO()
        joblib.dump(scaler_ml, scaler_buffer)
        scaler_buffer.seek(0)

        col2.download_button(
            label="💾 Descargar K004_scaler_ml.joblib",
            data=scaler_buffer,
            file_name="K004_scaler_ml.joblib",
            mime="application/octet-stream",
            use_container_width=True,
            help="Descarga el modelo de escalado entrenado (guarda exclusivamente las reglas para las 5 variables físicas)."
        )



