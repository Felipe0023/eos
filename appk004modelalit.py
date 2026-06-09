import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def BLOQUE004():
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>🧬 Índices de Metales Pesados y Riesgo a la Salud</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Evaluación de HPI, MI, Cd y Riesgo Toxicológico (Adultos y Niños)</p>", unsafe_allow_html=True)
        
        # 1. Verificar si existen datos cargados en el session_state
        if "K001_datos" not in st.session_state or st.session_state["K001_datos"] is None:
            st.warning("⚠️ No se detectaron datos cargados. Por favor, sube un archivo en la pestaña correspondiente.")
            return

        # 2. Cargar datos (hacer copia para evitar SettingWithCopyWarning)
        df = st.session_state["K001_datos"].copy()
        df.columns = df.columns.str.strip() # Limpieza preventiva de espacios

        # Parámetros estándar de Metales (Límite_Si, Peso_wi, RfD_Oral)
        parametros_metales = {
            'Al': (0.2,   0.02, 1.0),
            'Cd': (0.003, 0.3,  0.0005),
            'Cr': (0.05,  0.02, 0.003),
            'Cu': (2.0,   0.001,0.04),
            'Fe': (0.3,   0.003,0.7),
            'Mn': (0.1,   0.01, 0.14),
            'Ni': (0.02,  0.05, 0.02),
            'Pb': (0.01,  0.1,  0.0035),
            'Zn': (3.0,   0.0003,0.30)
        }

        # Filtrar metales presentes
        metales_presentes = {k: v for k, v in parametros_metales.items() if k in df.columns}

        if len(metales_presentes) == 0:
            st.error("❌ El set de datos no contiene ninguna columna con los metales requeridos (Al, Cd, Cr, Cu, Fe, Mn, Ni, Pb, Zn).")
            return

        with st.expander("🔬 Ver parámetros de referencia toxicológica (US EPA)"):
            st.write("Metales detectados y evaluados:", list(metales_presentes.keys()))
            df_ref = pd.DataFrame.from_dict(metales_presentes, orient='index', columns=['Límite Permisible (Si)', 'Peso (wi)', 'RfD Oral'])
            st.table(df_ref)

        # =====================================================================
        # CÁLCULO DE LOS ÍNDICES (Fórmulas optimizadas)
        # =====================================================================
        hpi_list, mi_list, cd_list, hi_adulto_list, hi_nino_list = [], [], [], [], []

        # Parámetros fijos para la evaluación de riesgo (Adultos vs Niños)
        IR_adulto, BW_adulto = 2.0, 70.0   
        IR_nino, BW_nino = 1.5, 15.0  

        sum_wi_hpi = sum(v[1] for v in metales_presentes.values())

        for idx, row in df.iterrows():
            sum_qi_wi = 0
            mi_acumulado = 0
            cd_acumulado = 0
            hi_ad_acumulado = 0
            hi_ni_acumulado = 0

            for metal, (Si, wi, RfD) in metales_presentes.items():
                C = row[metal]
                if pd.isna(C):
                    continue

                # A) HPI
                qi = (C / Si) * 100
                sum_qi_wi += qi * wi

                # B) MI
                mi_acumulado += (C / Si)

                # C) Contamination Degree (Cd)
                cd_acumulado += ((C / Si) - 1)

                # D) Hazard Index (HI Adulto e Infantil)
                cdi_adulto = (C * IR_adulto) / BW_adulto
                cdi_nino = (C * IR_nino) / BW_nino
                
                hi_ad_acumulado += (cdi_adulto / RfD)
                hi_ni_acumulado += (cdi_nino / RfD)

            hpi_list.append(sum_qi_wi / sum_wi_hpi if sum_wi_hpi > 0 else np.nan)
            mi_list.append(mi_acumulado)
            cd_list.append(cd_acumulado)
            hi_adulto_list.append(hi_ad_acumulado)
            hi_nino_list.append(hi_ni_acumulado)

        df['HPI'] = hpi_list
        df['MI'] = mi_list
        df['Cd_Index'] = cd_list  
        df['HI_Adulto'] = hi_adulto_list
        df['HI_Nino'] = hi_nino_list

        # Clasificaciones cualitativas mapeadas
        def clasificar_indices(row):
            c_hpi = "Bajo (<100)" if row['HPI'] < 100 else "Crítico (≥100)"
            c_mi = "Limpio (<1)" if row['MI'] < 1 else ("Moderado (1-6)" if row['MI'] <= 6 else "Crítico (>6)")
            c_cd = "Baja (<1)" if row['Cd_Index'] < 1 else ("Media (1-3)" if row['Cd_Index'] <= 3 else "Alta (>3)") 
            c_hi_ad = "Seguro (≤1)" if row['HI_Adulto'] <= 1 else "Riesgo a la salud (>1)"
            c_hi_ni = "Seguro (≤1)" if row['HI_Nino'] <= 1 else "Riesgo a la salud (>1)"
            return pd.Series([c_hpi, c_mi, c_cd, c_hi_ad, c_hi_ni])

        columnas_clases = ['Clase_HPI', 'Clase_MI', 'Clase_Cd', 'Clase_HI_Adulto', 'Clase_HI_Nino']
        df[columnas_clases] = df.apply(clasificar_indices, axis=1)

        # Mapas de paletas discretas compartidas entre gráficos de barras y mapas
        paletas_colores = {
            'Clase_HPI': {"Bajo (<100)": "#3B82F6", "Crítico (≥100)": "#EF4444"},
            'Clase_MI': {"Limpio (<1)": "#10B981", "Moderado (1-6)": "#F59E0B", "Crítico (>6)": "#EF4444"},
            'Clase_Cd': {"Baja (<1)": "#10B981", "Media (1-3)": "#F59E0B", "Alta (>3)": "#EF4444"},
            'Clase_HI_Adulto': {"Seguro (≤1)": "#3B82F6", "Riesgo a la salud (>1)": "#7F1D1D"},
            'Clase_HI_Nino': {"Seguro (≤1)": "#3B82F6", "Riesgo a la salud (>1)": "#7F1D1D"}
        }

        # Guardar en memoria global de la App
        st.session_state["K001_datos"] = df

        # =====================================================================
        # SECCIÓN GRÁFICA INTERACTIVA (TABS INTERNOS)
        # =====================================================================
        st.write("---")
        st.markdown("#### 📊 Visualización de Resultados")
        
        tab_barras, tab_mapas = st.tabs(["📊 Gráficos de Distribución (Barras)", "🗺️ Mapeo Geoespacial de Índices"])

        with tab_barras:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hpi = px.bar(df['Clase_HPI'].value_counts().reset_index(), x='Clase_HPI', y='count', 
                                 title="Distribución de Clases HPI", labels={'count': 'Muestras', 'Clase_HPI':'Clase'},
                                 color='Clase_HPI', color_discrete_map=paletas_colores['Clase_HPI'])
                st.plotly_chart(fig_hpi, use_container_width=True)
                
                fig_cd = px.bar(df['Clase_Cd'].value_counts().reset_index(), x='Clase_Cd', y='count', 
                                title="Grado de Contaminación (Cd Index)", labels={'count': 'Muestras', 'Clase_Cd':'Clase'},
                                color='Clase_Cd', color_discrete_map=paletas_colores['Clase_Cd'])
                st.plotly_chart(fig_cd, use_container_width=True)

                fig_hi_ni = px.bar(df['Clase_HI_Nino'].value_counts().reset_index(), x='Clase_HI_Nino', y='count', 
                                title="Índice de Peligro a la Salud (HI Niños)", labels={'count': 'Muestras', 'Clase_HI_Nino':'Clase'},
                                color='Clase_HI_Nino', color_discrete_map=paletas_colores['Clase_HI_Nino'])
                st.plotly_chart(fig_hi_ni, use_container_width=True)

            with col2:
                fig_mi = px.bar(df['Clase_MI'].value_counts().reset_index(), x='Clase_MI', y='count', 
                                title="Índice de Metales (MI)", labels={'count': 'Muestras', 'Clase_MI':'Clase'},
                                color='Clase_MI', color_discrete_map=paletas_colores['Clase_MI'])
                st.plotly_chart(fig_mi, use_container_width=True)
                
                fig_hi_ad = px.bar(df['Clase_HI_Adulto'].value_counts().reset_index(), x='Clase_HI_Adulto', y='count', 
                                title="Índice de Peligro a la Salud (HI Adulto)", labels={'count': 'Muestras', 'Clase_HI_Adulto':'Clase'},
                                color='Clase_HI_Adulto', color_discrete_map=paletas_colores['Clase_HI_Adulto'])
                st.plotly_chart(fig_hi_ad, use_container_width=True)

        with tab_mapas:
            if 'Latitud' in df.columns and 'Longitud' in df.columns:
                df_mapa = df.dropna(subset=['Latitud', 'Longitud'])
                
                cm1, cm2 = st.columns(2)
                with cm1:
                    tipo_mapa = st.radio("Tipo de visualización espacial:", ["Por Clases Cualitativas", "Escala Continua (Numérica)"])
                with cm2:
                    if tipo_mapa == "Por Clases Cualitativas":
                        clase_seleccionada = st.selectbox("Selecciona la clase a mapear:", 
                                                          ['Clase_HPI', 'Clase_MI', 'Clase_Cd', 'Clase_HI_Adulto', 'Clase_HI_Nino'])
                    else:
                        indice_seleccionado = st.selectbox("Selecciona el índice numérico a mapear:", 
                                                           ['HPI', 'MI', 'Cd_Index', 'HI_Adulto', 'HI_Nino'])
                
                columnas_id = [c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df.columns]
                id_hover = columnas_id[0] if columnas_id else df.columns[0]

                # RENDERIZADO DEL MAPA SEGÚN EL TIPO ELEGIDO
                if tipo_mapa == "Por Clases Cualitativas":
                    # Ordenación forzada de categorías en leyenda para consistencia visual
                    categoria_orden = list(paletas_colores[clase_seleccionada].keys())
                    
                    fig_2d = px.scatter_mapbox(
                        df_mapa, lat="Latitud", lon="Longitud",
                        color=clase_seleccionada,
                        color_discrete_map=paletas_colores[clase_seleccionada],
                        category_orders={clase_seleccionada: categoria_orden},
                        hover_name=id_hover,
                        hover_data={"Latitud": False, "Longitud": False, clase_seleccionada: True},
                        title=f"Mapeo de Zonas por Riesgo Ambiental: {clase_seleccionada.replace('_',' ')}"
                    )
                else:
                    fig_2d = px.scatter_mapbox(
                        df_mapa, lat="Latitud", lon="Longitud",
                        color=indice_seleccionado,
                        color_continuous_scale=["#0000FF", "#3B82F6", "#F59E0B", "#FF0000"],
                        range_color=[0, max(1.5 if 'HI' in indice_seleccionado else 120, df_mapa[indice_seleccionado].max())],
                        hover_name=id_hover,
                        hover_data={indice_seleccionado: ":.3f", "Latitud": False, "Longitud": False},
                        title=f"Distribución Espacial Continua del Índice: {indice_seleccionado}"
                    )

                # Ajustes globales de diseño del mapa (Duplicado de tamaño + Ajuste de límites)
                fig_2d.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        bounds={
                            "west": df_mapa["Longitud"].min() - 0.04, 
                            "east": df_mapa["Longitud"].max() + 0.04, 
                            "south": df_mapa["Latitud"].min() - 0.04, 
                            "north": df_mapa["Latitud"].max() + 0.04
                        }
                    ),
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                
                # Forzar el doble del tamaño de los puntos de forma segura
                fig_2d.update_traces(marker=dict(size=28))
                
                st.plotly_chart(fig_2d, use_container_width=True, config={'mapboxAccessToken': ''})
            else:
                st.info("ℹ️ Para proyectar los mapas geoespaciales, el archivo CSV original debe contener las columnas 'Latitud' y 'Longitud'.")

        # =====================================================================
        # MATRIZ DE DATOS Y ENLACE DE DESCARGA
        # =====================================================================
        st.write("---")
        st.markdown("##### 📋 Matriz de Resultados de Evaluación Ambiental")
        
        columnas_id = [c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df.columns] 
        id_col = columnas_id[0] if columnas_id else df.columns[0]
        
        columnas_finales = [
            id_col, 
            'HPI', 'Clase_HPI', 
            'MI', 'Clase_MI', 
            'Cd_Index', 'Clase_Cd', 
            'HI_Adulto', 'Clase_HI_Adulto',
            'HI_Nino', 'Clase_HI_Nino'
        ] + list(metales_presentes.keys())
        
        st.dataframe(df[columnas_finales], use_container_width=True, hide_index=True)

        # Botón de Descarga
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte Toxicológico Completo (CSV)",
            data=csv_buffer,
            file_name="datos_evaluacion_metales.csv",
            mime="text/csv",
            use_container_width=True
        )

