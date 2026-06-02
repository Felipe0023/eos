import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import rasterio
from io import BytesIO

def BLOQUE001(K001_dem):
    st.markdown("<h4 style='text-align: center;'>Generación de Malla 3D para Datos Nuevos</h4>", unsafe_allow_html=True)
    
    # 1. VERIFICAR QUE EL DEM EXISTE EN LOS BYTES PASADOS
    if K001_dem is None:
        st.warning("⚠️ No se han encontrado los datos del archivo TIF (DEM). Asegúrate de cargarlo en la barra lateral.")
        return

    # 2. CONFIGURACIÓN DE PARÁMETROS
    with st.expander("⚙️ Parámetros de Extrusión y Resolución Espacial", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        paso_xy_metros = col1.number_input("Paso XY (Metros en terreno)", min_value=500.0, value=500.0, step=10.0,
                                           help="Resolución horizontal en metros. Tu DEM base es de 30m.")
        
        paso_z = col2.number_input("Paso Z (Metros verticales)", min_value=0.5, value=10.0, step=0.5)
        distancia_total = col3.number_input("Profundidad total (Metros)", min_value=1.0, value=50.0, step=10.0)
        
        num_capas_z = int(np.ceil(distancia_total / paso_z))

    # 3. PROCESAMIENTO DE LA MALLA
    if st.button("🚀 Generar Malla de Subsuelo"):
        with st.spinner("Leyendo matriz GeoTIFF, aplicando grilla espacial y extruyendo capas..."):
            try:
                # Conversión de los bytes en memoria a un Dataset de Rasterio
                with rasterio.open(BytesIO(K001_dem)) as src:
                    banda1 = src.read(1) # Leemos la primera banda de altitudes
                    
                    # Generar matrices de coordenadas de píxeles
                    filas, columnas = banda1.shape
                    cols_idx, filas_idx = np.meshgrid(np.arange(columnas), np.arange(filas))
                    
                    # Transformar índices de píxel a coordenadas reales del mapa (X, Y)
                    xs, ys = rasterio.transform.xy(src.transform, filas_idx, cols_idx)
                    
                    # Convertir a vectores unidimensionales y aplanar
                    longitudes = np.array(xs).flatten()
                    latitudes = np.array(ys).flatten()
                    altitudes = banda1.flatten()
                    
                    # Crear DataFrame base
                    df_tif = pd.DataFrame({
                        'Longitud': longitudes,
                        'Latitud': latitudes,
                        'Altitud': altitudes
                    })
                    
                    # Limpieza de valores nulos o NoData del raster
                    if src.nodata is not None:
                        df_tif = df_tif[df_tif['Altitud'] != src.nodata]
                    df_tif = df_tif.dropna()

                # DETECTAR SI ES UTM (Metros) O GEOGRÁFICAS (Grados)
                es_utm = df_tif['Longitud'].max() > 180 or df_tif['Longitud'].min() < -180
                
                # --- LÓGICA DE FILTRADO ESPACIAL ---
                if es_utm:
                    df_filtrado = df_tif.groupby([
                        np.round(df_tif['Longitud'] / paso_xy_metros) * paso_xy_metros,
                        np.round(df_tif['Latitud'] / paso_xy_metros) * paso_xy_metros
                    ]).first().reset_index(drop=True)
                else:
                    # 1m aprox = 1/111000 grados
                    paso_grados = paso_xy_metros / 111000.0
                    df_filtrado = df_tif.groupby([
                        np.round(df_tif['Longitud'] / paso_grados) * paso_grados,
                        np.round(df_tif['Latitud'] / paso_grados) * paso_grados
                    ]).first().reset_index(drop=True)
                
                # Construcción optimizada de la malla 3D mediante matrices
                pts = []
                for _, fila in df_filtrado.iterrows():
                    x = fila['Longitud']
                    y = fila['Latitud']
                    z_superficie = fila['Altitud']
                    
                    for i in range(num_capas_z):
                        prof_actual = i * paso_z
                        pts.append({
                            'Profundidad': prof_actual,
                            'Longitud': round(x, 5),
                            'Latitud': round(y, 5),
                            'Altitud': z_superficie - prof_actual
                        })
                
                df_resultado = pd.DataFrame(pts)
                df_resultado['Cota'] = df_resultado['Profundidad'] + df_resultado['Altitud']
                
                # --- CAMBIO DE NOMBRE EN ALMACENAMIENTO (SALIDA) ---
                st.session_state['K008_Datos_Nuevos'] = df_resultado
                
                st.success(f"✅ Malla construida con éxito. Puntos en superficie: {len(df_filtrado)} | Total 3D: {len(df_resultado):,}")
            
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo TIF: {e}")

    # 4. VISUALIZACIÓN Y DESCARGA
    # Verificación corregida a la nueva nomenclatura de sesión
    if 'K008_Datos_Nuevos' in st.session_state:
        df_nuevos = st.session_state['K008_Datos_Nuevos']

        st.markdown("---")
        with st.container(border=True):
            st.markdown("#### 🧊 Visualización del Modelo de Bloques")
        
            # Muestra aleatoria para no colapsar la GPU/renderizado de Plotly
            df_plot = df_nuevos.sample(n=min(20000, len(df_nuevos))) if len(df_nuevos) > 20000 else df_nuevos

            fig = go.Figure(go.Scatter3d(
                x=df_plot['Longitud'],
                y=df_plot['Latitud'],
                z=df_plot['Altitud'],
                mode='markers',
                marker=dict(
                    size=2,
                    color=df_plot['Profundidad'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Profundidad (m)")
                )
            ))

            fig.update_layout(
                scene=dict(aspectratio=dict(x=1, y=1, z=0.4)),
                margin=dict(l=0, r=0, b=0, t=30),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)

            # 5. BOTÓN DE DESCARGA REESTRUCTURADO
            csv = df_nuevos.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar K008_Datos_Nuevos.csv",
                data=csv,
                file_name="K008_Datos_Nuevos.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("✅ Variable asignada y lista en memoria global de la app para su procesamiento en los siguientes bloques numéricos.")
