import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

def BLOQUE001(): # _PRONOSTICO LITOLOGICO
    st.markdown("<h4 style='text-align: center;'>Pronóstico Masivo de Tipo de Roca</h4>", unsafe_allow_html=True)
    st.info("Procesamiento automatizado del archivo de entrada 'K008_Datos_Nuevos' hacia 'K009_Nuevos_Datos'.")

    # 1. VERIFICAR ARTEFACTOS DE IA EN LA SESIÓN
    requisitos_ia = ['K004_scaler_ml', 'modelo_entrenado_roca', 'label_encoder_roca', 'asignador_hgs_knn']
    if any(req not in st.session_state for req in requisitos_ia):
        st.warning("⚠️ Faltan componentes de IA en la memoria del sistema.")
        st.info("Por favor, ejecuta primero el bloque de entrenamiento unificado (K005) para activar los modelos.")
        return

    # 2. ESCUDO DE DETECCIÓN EN MEMORIA
    nombre_clave_detectado = None
    variantes_nombre = ['K008_Datos_Nuevos', 'K_008_Datos_Nuevos.csv', 'K_008_datos_Nuevos.csv', 'K_008_Datos_Nuevos', 'K_008_datos_Nuevos']
    
    for variante in variantes_nombre:
        if variante in st.session_state and st.session_state[variante] is not None:
            nombre_clave_detectado = variante
            break

    if nombre_clave_detectado is None:
        st.error("❌ No se encontró el archivo de entrada en la sesión activa.")
        llaves_existentes = [llave for llave in st.session_state.keys() if '008' in llave or 'Datos' in llave or 'datos' in llave]
        if llaves_existentes:
            st.info(f"💡 Variables detectadas en memoria: `{llaves_existentes}`")
        return

    # Recuperamos los datos de la memoria
    df_nuevos_original = st.session_state[nombre_clave_detectado].copy()
    total_registros = len(df_nuevos_original)
    st.success(f"📊 Dataset de entrada detectado con éxito: `{nombre_clave_detectado}` ({total_registros} registros).")

    cols_requeridas = ['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota']
    if not all(col in df_nuevos_original.columns for col in cols_requeridas):
        st.error(f"❌ El archivo en memoria no cuenta con las columnas físicas necesarias: {cols_requeridas}")
        return

    st.write("**Vista previa de los datos de entrada (Escala Real):**")
    st.dataframe(df_nuevos_original.head(5), use_container_width=True)

    # --- BOTÓN DE CONTROL DE PROCESAMIENTO MASIVO ---
    if st.button("🔮 Ejecutar Pronóstico Geológico Masivo", use_container_width=True):
        df_proc = df_nuevos_original.copy()

        with st.spinner("⏳ Aplicando transformaciones matemáticas e inyectando HGS..."):
            try:
                # PASO 1: ESCALAMIENTO GLOBAL
                scaler_global = st.session_state['K004_scaler_ml']
                df_proc[cols_requeridas] = scaler_global.transform(df_proc[cols_requeridas])

                # PASO 2: ASIGNACIÓN DE HGS CONSISTENTE
                asignador_knn = st.session_state['asignador_hgs_knn']
                cols_coordenadas = ['Profundidad', 'Longitud', 'Latitud', 'Altitud']
                df_proc['HGS'] = asignador_knn.predict(df_proc[cols_coordenadas])
                df_proc['HGS'] = df_proc['HGS'].astype(int)
                
                # PASO 3: VARIABLE BINARIA DERIVADA
                df_proc['Es_Zona_Compleja'] = (df_proc['HGS'] == 999).astype(int)

                # PASO 4: ORDENAR MATRIZ X PARA EL MODELO XGBOOST
                features_modelo = ['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota', 'HGS', 'Es_Zona_Compleja']
                X_pred = df_proc[features_modelo]

                # PASO 5: CLASIFICACIÓN CON EL MODELO XGBOOST
                modelo_xgb = st.session_state['modelo_entrenado_roca']
                predicciones_num = modelo_xgb.predict(X_pred)

                # PASO 6: DESCODIFICACIÓN FINAL A ETIQUETAS LITOLÓGICAS
                label_encoder = st.session_state['label_encoder_roca']
                
                # Mapeamos los resultados estructurales sobre el DataFrame original
                df_nuevos_original['HGS'] = df_proc['HGS'] 
                df_nuevos_original['Es_Zona_Compleja'] = df_proc['Es_Zona_Compleja']
                
                # 🌟 CORRECCIÓN CRÍTICA: Cambiar 'Tipo_Roca_Predicho' por 'Tipo_Roca' 
                # para que coincida exactamente con las columnas numéricas/categóricas de K006
                df_nuevos_original['Tipo_Roca'] = label_encoder.inverse_transform(predicciones_num)

                # 🌟 SOLUCIÓN AL ERROR DE MEMORIA: Guardamos en ambas claves (con y sin extensión)
                st.session_state['K009_Nuevos_Datos'] = df_nuevos_original
                st.session_state['K009_Nuevos_Datos.csv'] = df_nuevos_original

                st.success(f"🎉 ¡Pronóstico completado exitosamente! El dataset 'K009_Nuevos_Datos' está listo para el modelo de regresión.")
                st.rerun() 
                
            except Exception as e:
                st.error(f"❌ Error crítico en la ejecución del pipeline: {e}")
                return

    # 3. MUESTRA DE RESULTADOS Y EXPORTACIÓN
    if 'K009_Nuevos_Datos' in st.session_state and st.session_state['K009_Nuevos_Datos'] is not None:
        df_resultados = st.session_state['K009_Nuevos_Datos']
        
        st.write("**Resultados del Pronóstico (Primeros 10 registros con columna 'Tipo_Roca' unificada):**")
        st.dataframe(df_resultados.head(10), use_container_width=True)

        # --- PANEL DE DESCARGA ---
        with st.container(border=True):
            st.markdown("##### 📥 Exportar Resultados Litológicos")
            csv_data = df_resultados.to_csv(index=False).encode('utf-8-sig')

            st.download_button(
                label="📄 Descargar Dataset K009_Nuevos_Datos (.csv)",
                data=csv_data,
                file_name="K009_Nuevos_Datos.csv", 
                mime="text/csv",
                use_container_width=True
            )





