import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image 
import rasterio
import plotly.express as px
import pydeck as pdk
import os
import io
from io import BytesIO

def BLOQUE001(df_raw, tif_bytes):
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Localización de las Perforaciones</h3>", unsafe_allow_html=True)

        # --- 1. EXTRAER DATOS DEL TIF EN MEMORIA RAM ---
        try:
            with rasterio.open(io.BytesIO(tif_bytes)) as src:
                data_dem = src.read(1)  # Leer la primera banda (matriz de elevación)
                meta = src.meta
                bounds = src.bounds
        except Exception as e:
            st.error(f"❌ Error al decodificar el archivo TIF desde la memoria RAM: {e}")
            return
        
        # Convertir a float y manejar NoData
        z_data = data_dem.astype(float)
        nodata_val = meta.get('nodata')
        if nodata_val is not None:
            z_data[z_data == nodata_val] = np.nan

        # Generar coordenadas espaciales basadas en los límites reales del mapa
        x_coords = np.linspace(bounds.left, bounds.right, z_data.shape[1])
        y_coords = np.linspace(bounds.bottom, bounds.top, z_data.shape[0])

        # --- 2. MAPA PLOTLY ---
        fig_map = go.Figure()

        # Heatmap (Mapa de elevación/fondo)
        fig_map.add_trace(go.Heatmap(
            x=x_coords, 
            y=y_coords, 
            z=np.flipud(z_data), 
            colorscale='earth',
            name='Elevación',
            showscale=True
        ))

        # Scatter (Puntos de perforación)
        fig_map.add_trace(go.Scatter(
            x=df_raw['Longitud'], 
            y=df_raw['Latitud'], 
            mode='markers', 
            marker=dict(color='red', size=36, symbol='diamond', line=dict(width=1, color='white')),
            name='Perforaciones'
        ))

        # --- 3. DISEÑO (Layout) ---
        fig_map.update_layout(
            width=800, 
            height=600,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
                
        st.plotly_chart(fig_map, use_container_width=True)

        # 1. Guardar el gráfico en un buffer de memoria
        buffer = io.StringIO()
        fig_map.write_html(buffer, include_plotlyjs='cdn')
        html_bytes = buffer.getvalue().encode()

        # 2. Botón de descarga
        st.download_button(
            label="📥 Descargar Mapa HTML",
            data=html_bytes,
            file_name="Mapa_Localizacion_NEREUS.html",
            mime="text/html"
        )




def BLOQUE002(df_raw, mapbox_key):   
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Monitoreo de Perforaciones</h3>", unsafe_allow_html=True)
 
        # Fila de Métricas
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.metric("Total de Registros", len(df_raw))
        with c2: 
            st.metric("Altitud Promedio", f"{df_raw['Altitud'].mean():.1f} m")
        with c3: 
            st.metric("Profundidad Promedio", f"{df_raw['Profundidad'].mean():.1f} m")
        
        # 1. Mantenemos tu función de color basada en la Profundidad
        def color_por_profundidad(prof):
            try:
                v = float(prof)
                intensidad = min(255, int(abs(v) * 2.5))
                return [intensidad, 100, 255 - intensidad, 180]
            except:
                return [0, 150, 255, 180] # Celeste por defecto si falla

        # 2. Verificamos las columnas de posicionamiento necesarias
        columnas_requeridas = ['Longitud', 'Latitud', 'Profundidad', 'Cota', 'Altitud']
        if all(col in df_raw.columns for col in ['Longitud', 'Latitud']):
            df_mapa = df_raw.copy()
            df_mapa['color'] = df_mapa['Profundidad'].apply(color_por_profundidad)
            
            # 💡 CONFIGURACIÓN PARA EL ICONO CUADRADO:
            # Usamos un archivo SVG/PNG transparente de un cuadrado perfecto
            ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/d/de/Square_Carbon_Solid.png"
            
            # Definimos la estructura del icono para PyDeck
            icon_data = {
                "url": ICON_URL,
                "width": 128,
                "height": 128,
                "anchorY": 64, # Centrado vertical
                "anchorX": 64, # Centrado horizontal
                "mask": True   # Permite que PyDeck le pinte encima tu función de colores dinámicos
            }
            
            # Asignamos el diccionario del icono a cada fila del mapa
            df_mapa["icon_data"] = [icon_data] * len(df_mapa)
                
            # Tooltip con recuadro formateado solo con las 5 variables requeridas
            t_html = """
            <div style="
                background-color: #ffffff; 
                color: #333333; 
                padding: 12px; 
                border-radius: 8px; 
                border: 1px solid #cccccc;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                font-family: sans-serif;
                line-height: 1.5;
            ">
                <b style="color: #1E3A8A;">📍 Longitud:</b> {Longitud}<br>
                <b style="color: #1E3A8A;">📍 Latitud:</b> {Latitud}<br>
                <hr style="margin: 4px 0; border: 0; border-top: 1px solid #eee;">
                <b style="color: #0284C7;">📐 Profundidad:</b> {Profundidad} m<br>
                <b style="color: #0284C7;">🏔️ Cota:</b> {Cota} m<br>
                <b style="color: #0284C7;">🏔️ Altitud:</b> {Altitud} m
            </div>
            """

            # 3. Renderizar Mapa mediante PyDeck
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/outdoors-v12',
                api_keys={'mapbox': mapbox_key},
                initial_view_state=pdk.ViewState(
                    latitude=df_mapa["Latitud"].mean(),
                    longitude=df_mapa["Longitud"].mean(),
                    zoom=11, 
                    pitch=45
                ),
                layers=[
                    # 💡 SE CAMBIÓ CAMBIÓ DE "ScatterplotLayer" A "IconLayer"
                    pdk.Layer(
                        "IconLayer",
                        df_mapa,
                        get_position=["Longitud", "Latitud"],
                        get_icon="icon_data",
                        get_color="color",
                        pickable=True,
                        # 💡 DUPLICADO DE TAMAÑO: Pasó de un equivalente visual de 12 a un tamaño de 24
                        get_size=4, 
                        size_scale=15,
                        size_units="'meters'"
                    )
                ],
                tooltip={"html": t_html}
            ))
        else:
            st.error("⚠️ Faltan columnas de posicionamiento geográfico (Latitud/Longitud) en el archivo de datos.")












def BLOQUE003(df_raw, df_tif, submuestreo=5):
    with st.container(border=True):
        # 1. Validaciones iniciales
        if df_raw is None or df_tif is None:
            st.error("❌ Faltan datos (CSV o TIF) para generar el mapa 3D.")
            return

        # --- 2. PROCESAR DATOS DEL DATAFRAME (Puntos de perforación) ---
        df = df_raw.copy()
        columnas_necesarias = ['Longitud', 'Latitud', 'Altitud']
        
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"❌ El CSV debe tener las columnas: {columnas_necesarias}")
            return

        puntos_lon = df['Longitud'].values
        puntos_lat = df['Latitud'].values
        puntos_alt = df['Altitud'].values

        # --- 3. PROCESAR DATOS DEL TIF (Superficie 3D desde Bytes) ---
        try:
            with rasterio.open(BytesIO(df_tif)) as src:
                data_dem = src.read(1)  # Leemos la primera banda del raster
                meta = src.meta.copy()
                bounds = src.bounds
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo TIF en el modelo 3D: {e}")
            return

        # Aplicamos submuestreo directamente a la matriz en memoria
        # [::submuestreo, ::submuestreo] reduce la resolución para que el 3D sea fluido
        z_data = data_dem[::submuestreo, ::submuestreo].astype(float)
        
        # Manejo de valores NoData usando los metadatos guardados
        nodata_val = meta.get('nodata')
        if nodata_val is not None:
            z_data[z_data == nodata_val] = np.nan
        
        # Generar mallas de coordenadas basadas en los límites geográficos
        x_coords = np.linspace(bounds.left, bounds.right, z_data.shape[1])
        y_coords = np.linspace(bounds.bottom, bounds.top, z_data.shape[0])

        # --- 4. CREAR FIGURA PLOTLY 3D ---
        fig = go.Figure()
        
        # Trace de la superficie (Terreno)
        fig.add_trace(go.Surface(
            x=x_coords, 
            y=y_coords, 
            z=z_data, 
            colorscale='earth', 
            showscale=False, 
            name='Terreno'
        ))
        
        # Trace de los puntos (Perforaciones)
        fig.add_trace(go.Scatter3d(
            x=puntos_lon, 
            y=puntos_lat, 
            z=puntos_alt, 
            mode='markers', 
            marker=dict(size=4, color='red', opacity=0.8), 
            name='Puntos'
        ))

        # Ajustes de diseño para mejorar la perspectiva
        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=0.3), # Exageración vertical para notar desniveles
                zaxis_title='Altitud (m)',
                xaxis_title='Longitud',
                yaxis_title='Latitud'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # --- 5. RENDERIZADO ---
        st.markdown("<h4 style='text-align: center;'>Modelo Digital de Elevación 3D</h4>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- TABLA Y DESCARGA ---
        with st.expander("🔍 Ver tabla de datos"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        # Botones de descarga de reportes
        col_html, col_csv = st.columns(2)

        with col_html:        
            html_bytes = fig.to_html().encode('utf-8')
            st.download_button(
                label="📥 Descargar Mapa HTML",
                data=html_bytes,
                file_name="NEREUS_Mapa_3D.html",
                mime="text/html"
            )

        with col_csv:        
            csv_data = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar K_001_Datos.csv",
                data=csv_data,
                file_name='K_001_Datos.csv',
                mime='text/csv'
            )
       
        # --- 6. GUARDAR PARA EL SIGUIENTE NIVEL (K_002) ---
        st.session_state['K_001_Datos'] = df_raw.copy()
        st.toast("Datos listos para K_002", icon="💾")
        st.caption("✅ Datos sincronizados internamente para Clustering HGS.")
        
        return fig


