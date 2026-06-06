import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def BLOQUE003():
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>🚰 Evaluación de Calidad del Agua (DWQI)</h3>", unsafe_allow_html=True)
        
        # 1. Verificar si existen datos cargados en el session_state
        if "K001_datos" not in st.session_state or st.session_state["K001_datos"] is None:
            st.warning("⚠️ No se detectaron datos cargados. Por favor, sube un archivo en la pestaña correspondiente.")
            return

        # 2. Cargar los datos desde el session_state (hacer copia para evitar Warnings de asignación)
        df = st.session_state["K001_datos"].copy()
        
        # Limpieza preventiva contra espacios ocultos en los nombres de las columnas
        df.columns = df.columns.str.strip()

        # 3. Definir los Estándares de Referencia (Si), Pesos (wi) y Valores Ideales (Vi)
        parametros_calidad = {
            'pH': (8.5, 4, 7.0),
            'EC': (1500.0, 3, 0.0),
            'TDS': (1000.0, 3, 0.0),
            'Cl-': (250.0, 3, 0.0),
            'SO42-': (250.0, 3, 0.0),
            'NO3-': (50.0, 5, 0.0),
            'Fe': (0.3, 4, 0.0),
            'Mn': (0.1, 4, 0.0),
            'Pb': (0.01, 5, 0.0),
            'Cd': (0.003, 5, 0.0),
            'As': (0.01, 5, 0.0),
            'Zn': (3.0, 2, 0.0),
            'Cu': (2.0, 2, 0.0),
            'Al': (0.2, 3, 0.0),
            'Cr': (0.05, 5, 0.0)
        }

        # Filtrar solo los parámetros que realmente existen en el DataFrame
        param_disponibles = {k: v for k, v in parametros_calidad.items() if k in df.columns}

        if len(param_disponibles) == 0:
            st.error("❌ El set de datos no contiene variables hidroquímicas válidas para el cálculo del DWQI (pH, EC, TDS, etc.).")
            return

        # 4. Calcular constante de proporcionalidad (K) y Pesos Unitarios (Wi)
        sum_inv_si = sum(1 / valores[0] for valores in param_disponibles.values())
        K_val = 1 / sum_inv_si

        pesos_unitarios = {param: (K_val / valores[0]) for param, valores in param_disponibles.items()}
        Suma_Wi = sum(pesos_unitarios.values())

        # 5. Función optimizada por fila para calcular el sub-índice (qi) y la sumatoria
        def calcular_dwqi_fila(row):
            suma_qi_Wi = 0
            for param, (Si, wi, Vi) in param_disponibles.items():
                Xi = row[param]

                if pd.isna(Xi):
                    continue

                if param == 'pH':
                    qi = ((Xi - 7.0) / (Si - 7.0)) * 100
                else:
                    qi = (Xi / Si) * 100

                qi = max(0, qi)
                suma_qi_Wi += qi * pesos_unitarios[param]

            return suma_qi_Wi / Suma_Wi if Suma_Wi > 0 else np.nan

        # 6. Aplicar el cálculo del índice analítico
        df['DWQI'] = df.apply(calcular_dwqi_fila, axis=1)

        # 7. Clasificar rangos cualitativos de calidad
        def clasificar_calidad(dwqi):
            if pd.isna(dwqi): return 'Sin Datos'
            if dwqi <= 25: return 'Excelente'
            elif dwqi <= 50: return 'Buena'
            elif dwqi <= 75: return 'Pobre (Regular)'
            elif dwqi <= 100: return 'Muy Pobre'
            else: return 'No apta para consumo (Inadecuada)'

        df['Clasificacion_Calidad'] = df['DWQI'].apply(clasificar_calidad)

        # 🌟 CRÍTICO: Guardar los nuevos cálculos de vuelta en el session_state original 
        # para que estén disponibles globalmente en toda la aplicación
        st.session_state["K001_datos"] = df

        # ==================================================================================================================================
        # DESPLIEGUE EN INTERFAZ GRÁFICA DE STREAMLIT
        # ==================================================================================================================================
        
        # Bloque de Métricas Clave
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total de Muestras", len(df))
        with c2:
            st.metric("DWQI Promedio", f"{df['DWQI'].mean():.2f}")
        with c3:
            muestras_aptas = df['Clasificacion_Calidad'].isin(['Excelente', 'Buena']).sum()
            pct_aptas = (muestras_aptas / len(df)) * 100 if len(df) > 0 else 0
            st.metric("% Muestras Aceptables", f"{pct_aptas:.1f}%")

        # Gráfico dinámico de distribución mediante Plotly
        st.write("---")
        orden_categorias = ['Excelente', 'Buena', 'Pobre (Regular)', 'Muy Pobre', 'No apta para consumo (Inadecuada)']
        conteo = df['Clasificacion_Calidad'].value_counts().reindex(orden_categorias, fill_value=0).reset_index()
        conteo.columns = ['Condición', 'Cantidad']
        
        fig = px.bar(
            conteo, x='Condición', y='Cantidad', color='Condición',
            title="Distribución del Estatus de Calidad del Agua",
            color_discrete_map={
                'Excelente': '#1E3A8A', 'Buena': '#3B82F6', 
                'Pobre (Regular)': '#F59E0B', 'Muy Pobre': '#EF4444', 
                'No apta para consumo (Inadecuada)': '#7F1D1D'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla interactiva de resultados
        st.write("---")
        st.markdown("##### 📋 Matriz de Resultados Calculados")
        
        columnas_id = [c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df.columns]
        id_col = columnas_id[0] if columnas_id else df.columns[0]
        
        columnas_mostrar = [id_col, 'DWQI', 'Clasificacion_Calidad'] + list(param_disponibles.keys())
        st.dataframe(df[columnas_mostrar], use_container_width=True, hide_index=True)

        # Botón para que el usuario descargue la base de datos procesada
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Base de Datos con DWQI (CSV)",
            data=csv_buffer,
            file_name="datos_procesados_DWQI.csv",
            mime="text/csv",
            use_container_width=True
        )