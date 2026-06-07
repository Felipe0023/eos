import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def BLOQUE004():
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>🧬 Índices de Metales Pesados y Riesgo a la Salud</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Evaluación de HPI, MI, Cd y Riesgo Toxicológico (HI Adulto)</p>", unsafe_allow_html=True)
        
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
        hpi_list, mi_list, cd_list, hi_list = [], [], [], []

        # Parámetros fijos para el HI (Adultos)
        IR = 2.0   
        BW = 70.0  

        sum_wi_hpi = sum(v[1] for v in metales_presentes.values())

        for idx, row in df.iterrows():
            sum_qi_wi = 0
            mi_acumulado = 0
            cd_acumulado = 0
            hi_acumulado = 0

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

                # D) Hazard Index (HI)
                cdi = (C * IR) / BW
                hq = cdi / RfD
                hi_acumulado += hq

            hpi_list.append(sum_qi_wi / sum_wi_hpi if sum_wi_hpi > 0 else np.nan)
            mi_list.append(mi_acumulado)
            cd_list.append(cd_acumulado)
            hi_list.append(hi_acumulado)

        df['HPI'] = hpi_list
        df['MI'] = mi_list
        df['Cd_Index'] = cd_list  # 🛠️ CAMBIO 1: Renombrado para evitar duplicidad con el metal Cadmio
        df['HI_Adulto'] = hi_list

        # Clasificaciones cualitativas
        def clasificar_indices(row):
            c_hpi = "Bajo (<100)" if row['HPI'] < 100 else "Crítico (≥100)"
            c_mi = "Limpio (<1)" if row['MI'] < 1 else ("Moderado (1-6)" if row['MI'] <= 6 else "Crítico (>6)")
            c_cd = "Baja (<1)" if row['Cd_Index'] < 1 else ("Media (1-3)" if row['Cd_Index'] <= 3 else "Alta (>3)") # 🛠️ CAMBIO 2: Uso del nuevo nombre
            c_hi = "Seguro (≤1)" if row['HI_Adulto'] <= 1 else "Riesgo a la salud (>1)"
            return pd.Series([c_hpi, c_mi, c_cd, c_hi])

        df[['Clase_HPI', 'Clase_MI', 'Clase_Cd', 'Clase_HI']] = df.apply(clasificar_indices, axis=1)

        # Guardar en memoria global de la App
        st.session_state["K001_datos"] = df

        # =====================================================================
        # SECCIÓN GRÁFICA INTERACTIVA (TABS INTERNOS)
        # =====================================================================
        st.write("---")
        st.markdown("#### 📊 Visualización de Resultados")
        
        tab_barras, tab_mapas = st.tabs(["📊 Gráficos de Distribución (Barras)", "🗺️ Mapeo Geoespacial 2D"])

        with tab_barras:
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. HPI
                fig_hpi = px.bar(df['Clase_HPI'].value_counts().reset_index(), x='Clase_HPI', y='count', 
                                 title="Distribución de Clases HPI", labels={'count': 'Muestras', 'Clase_HPI':'Clase'},
                                 color='Clase_HPI', color_discrete_map={"Bajo (<100)": "#3B82F6", "Crítico (≥100)": "#EF4444"})
                st.plotly_chart(fig_hpi, use_container_width=True)
                
                # 2. Cd
                fig_cd = px.bar(df['Clase_Cd'].value_counts().reset_index(), x='Clase_Cd', y='count', 
                                title="Grado de Contaminación (Cd Index)", labels={'count': 'Muestras', 'Clase_Cd':'Clase'},
                                color='Clase_Cd', color_discrete_map={"Baja (<1)": "#10B981", "Media (1-3)": "#F59E0B", "Alta (>3)": "#EF4444"})
                st.plotly_chart(fig_cd, use_container_width=True)

            with col2:
                # 3. MI
                fig_mi = px.bar(df['Clase_MI'].value_counts().reset_index(), x='Clase_MI', y='count', 
                                title="Índice de Metales (MI)", labels={'count': 'Muestras', 'Clase_MI':'Clase'},
                                color='Clase_MI', color_discrete_map={"Limpio (<1)": "#10B981", "Moderado (1-6)": "#F59E0B", "Crítico (>6)": "#EF4444"})
                st.plotly_chart(fig_mi, use_container_width=True)
                
                # 4. HI Adulto
                fig_hi = px.bar(df['Clase_HI'].value_counts().reset_index(), x='Clase_HI', y='count', 
                                title="Índice de Peligro a la Salud (HI Adulto)", labels={'count': 'Muestras', 'Clase_HI':'Clase'},
                                color='Clase_HI', color_discrete_map={"Seguro (≤1)": "#3B82F6", "Riesgo a la salud (>1)": "#7F1D1D"})
                st.plotly_chart(fig_hi, use_container_width=True)

        with tab_mapas:
            if 'Latitud' in df.columns and 'Longitud' in df.columns:
                df_mapa = df.dropna(subset=['Latitud', 'Longitud'])
                
                # Selector dinámico
                indice_seleccionado = st.selectbox("Selecciona el índice numérico a graficar en el mapa:", 
                                                   ['HPI', 'MI', 'Cd_Index', 'HI_Adulto']) # 🛠️ CAMBIO 3: Cambiado 'Cd' por 'Cd_Index'
                
                columnas_id = [c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df.columns]
                id_hover = columnas_id[0] if columnas_id else df.columns[0]

                fig_2d = px.scatter_mapbox(
                    df_mapa, lat="Latitud", lon="Longitud",
                    color=indice_seleccionado,
                    size=np.repeat(14, len(df_mapa)),
                    color_continuous_scale=["#0000FF", "#3B82F6", "#F59E0B", "#FF0000"],
                    hover_name=id_hover,
                    hover_data={indice_seleccionado: ":.3f", "Clase_"+indice_seleccionado.split('_')[0]: True, "Latitud": False, "Longitud": False},
                    zoom=11,
                    title=f"Distribución Espacial Continua de {indice_seleccionado}"
                )

                fig_2d.update_layout(
                    mapbox=dict(style="open-street-map"),
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig_2d, use_container_width=True, config={'mapboxAccessToken': ''})
            else:
                st.info("ℹ️ Para proyectar los mapas 2D continuos, el archivo CSV original debe contener las columnas de georreferenciación 'Latitud' y 'Longitud'.")

        # =====================================================================
        # MATRIZ DE DATOS Y ENLACE DE DESCARGA
        # =====================================================================
        st.write("---")
        st.markdown("##### 📋 Matriz de Resultados de Evaluación Ambiental")
        
        columnas_id = [c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df.columns]
        id_col = columnas_id[0] if columnas_id else df.columns[0]
        
        # 🛠️ CAMBIO 4: Cambiado 'Cd' por 'Cd_Index' en la estructura final de columnas a imprimir
        columnas_finales = [id_col, 'HPI', 'Clase_HPI', 'MI', 'Clase_MI', 'Cd_Index', 'Clase_Cd', 'HI_Adulto', 'Clase_HI'] + list(metales_presentes.keys())
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




