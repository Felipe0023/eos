import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# DEFINICIÓN DE LA ARQUITECTURA DE LA RED DE ELMAN (Adaptada a Cd Index)
# =====================================================================
class ElmanRNNCdIndex(nn.Module):
    def __init__(self, input_size=20, hidden_size=144, output_size=1):
        super(ElmanRNNCdIndex, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# =====================================================================
# FUNCIONES AUXILIARES DE MODELADO MATEMÁTICO Y MÉTRICAS
# =====================================================================
def calcular_metricas_cd_index(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    nse = r2 
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mad = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mad, nse

def pronosticar_nuevos_datos_cd_index(modelo_entrenado, escalador_X, escalador_y, datos_nuevos_raw, device):
    modelo_entrenado.eval() 
    datos_array = np.array(datos_nuevos_raw, dtype=np.float32)
    if datos_array.ndim == 1:
        datos_array = datos_array.reshape(1, -1)
        
    datos_escalados = escalador_X.transform(datos_array)
    tensor_input = torch.tensor(datos_escalados, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        prediccion_tensor_scaled = modelo_entrenado(tensor_input)
        
    cd_index_pronosticado = escalador_y.inverse_transform(prediccion_tensor_scaled.cpu().numpy())
    return cd_index_pronosticado.flatten()

# =====================================================================
# 🌟 FUNCIÓN PRINCIPAL DEL MÓDULO PARA STREAMLIT
# =====================================================================
def MODULO_ENTRENAMIENTO_Y_PRONOSTICO_CD_INDEX():
    """
    Módulo IA homólogo diseñado exclusivamente para predecir el Índice de Contaminación (Cd Index)
    utilizando la matriz completa de 20 predictores ambientales.
    """
    st.markdown("<h3 style='text-align: center;'>🧠 Modelado de Inteligencia Artificial para Cd Index (Elman RNN)</h3>", unsafe_allow_html=True)
    
    # 1. Validación de dependencias en session_state (debe provenir del bloque matemático)
    if "K001_datos" not in st.session_state or st.session_state["K001_datos"] is None:
        st.warning("⚠️ No se detectó la matriz base con los índices calculados. Por favor, procese la pestaña de Índices de Metales Pesados primero.")
        return

    # Recuperar datos históricos
    df_historico = st.session_state["K001_datos"].copy()

    # Dispositivo de cálculo hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Definición de los 20 predictores (Se usan las 15 variables químicas incluyendo Cd y las 5 estructurales)
    columnas_predictores = [
        'pH', 'EC', 'TDS', 'Cl-', 'SO42-', 'NO3-', 'Fe', 'Mn', 'Pb', 'Cd', 'As', 'Zn', 'Cu', 'Al', 'Cr',
        'Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota'
    ]
    
    # Validar consistencia de columnas y del target Cd_Index / Cd_index
    columnas_faltantes = [col for col in columnas_predictores if col not in df_historico.columns]
    if columnas_faltantes:
        st.error(f"❌ Matriz incompleta en la sesión actual. Faltan las variables: {columnas_faltantes}")
        return
    
    # Manejar nombres alternos del target en el dataframe
    target_col = None
    for posible_nombre in ['Cd_Index', 'Cd_index', 'CD_INDEX', 'C_d']:
        if posible_nombre in df_historico.columns:
            target_col = posible_nombre
            break
            
    if target_col is None:
        st.error("❌ No se encontró la columna objetivo del índice (Cd_Index o C_d) calculada en tu set de datos.")
        return
        
    X_datos = df_historico[columnas_predictores].values
    y_datos = df_historico[target_col].values.reshape(-1, 1)

    # 2. Interfaz de Control de Entrenamiento
    st.subheader("🏋️ Entrenamiento del Modelo Predictivo del Índice de Contaminación (C_d)")
    st.info(f"Hardware Activo: `{device.type.upper()}` | Características de entrada (Inputs): `20` | Muestras: `{X_datos.shape[0]}`")

    with st.expander("⚙️ Hiperparámetros de Ajuste de la Red (Cd Index)", expanded=False):
        lr = st.slider("Tasa de Aprendizaje (Learning Rate)", 0.001, 0.050, 0.005, step=0.001, key="lr_cd_idx")
        epochs = st.slider("Épocas de Entrenamiento", 50, 500, 150, step=25, key="epochs_cd_idx")
        hidden_size = st.slider("Unidades Ocultas (Capa de Contexto)", 32, 256, 144, step=16, key="hidden_cd_idx")

    if st.button("🚀 Entrenar Red Neuronal Recurrente para Cd Index", use_container_width=True):
        with st.spinner("Optimizando pesos de la red de Elman para predicción del grado de contaminación acumulado..."):
            np.random.seed(42)
            torch.manual_seed(42)

            # Split y Escalado
            X_train, X_val, y_train, y_val = train_test_split(X_datos, y_datos, test_size=0.30, random_state=42)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled = scaler_y.transform(y_val)

            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

            # Instanciación de la Red de Elman con input_size=20
            model = ElmanRNNCdIndex(input_size=20, hidden_size=hidden_size, output_size=1).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            progreso_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss_train = criterion(outputs, y_train_t)
                loss_train.backward()
                optimizer.step()

                progreso_bar.progress((epoch + 1) / epochs)
                if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_t)
                        loss_val = criterion(val_outputs, y_val_t)
                    status_text.text(f"Época [{epoch+1}/{epochs}] -> Loss Train: {loss_train.item():.4f} | Loss Val: {loss_val.item():.4f}")

            # Métricas finales de control analítico
            model.eval()
            with torch.no_grad():
                pred_train_scaled = model(X_train_t).cpu().numpy()
                pred_train = scaler_y.inverse_transform(pred_train_scaled).flatten()
                pred_val_scaled = model(X_val_t).cpu().numpy()
                pred_val = scaler_y.inverse_transform(pred_val_scaled).flatten()

            r2_tr, rmse_tr, mad_tr, nse_tr = calcular_metricas_cd_index(y_train.flatten(), pred_train)
            r2_va, rmse_va, mad_va, nse_va = calcular_metricas_cd_index(y_val.flatten(), pred_val)

            # Persistencia de los artefactos en el Session State
            st.session_state["cd_index_model_fitted"] = model
            st.session_state["cd_index_scaler_x"] = scaler_X
            st.session_state["cd_index_scaler_y"] = scaler_y
            
            st.success("🎉 ¡Modelo RNN ajustado para la predicción del Cd Index guardado en memoria!")

            # Matriz de validación matemática en Streamlit
            st.markdown("##### 📊 Matriz Comparativa de Validez Matemática (Cd Index)")
            tabla_metricas = pd.DataFrame({
                "Métrica": ["Coef. Determinación (R²)", "Raíz Error Cuadrático (RMSE)", "Desviación Absoluta Media (MAD)", "Eficiencia Nash-Sutcliffe (NSE)"],
                "Entrenamiento": [f"{r2_tr:.4f}", f"{rmse_tr:.4f}", f"{mad_tr:.4f}", f"{nse_tr:.4f}"],
                "Validación": [f"{r2_va:.4f}", f"{rmse_va:.4f}", f"{mad_va:.4f}", f"{nse_va:.4f}"]
            })
            st.dataframe(tabla_metricas, use_container_width=True, hide_index=True)

    # 3. Módulos Operativos de Pronóstico (Inferencia)
    if "cd_index_model_fitted" in st.session_state:
        st.write("---")
        st.subheader("🔮 Panel Operativo de Inferencias (Índice de Contaminación - C_d)")
        
        model = st.session_state["cd_index_model_fitted"]
        scaler_X = st.session_state["cd_index_scaler_x"]
        scaler_y = st.session_state["cd_index_scaler_y"]

        tab1, tab2 = st.tabs(["🕹️ Pronóstico Individual", "📂 Pronóstico Masivo (CSV)"])

        with tab1:
            st.markdown("##### Ingrese los parámetros de la matriz completa:")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                p_pH = st.number_input("pH", 0.0, 14.0, 7.2, key="cdidx_in_pH")
                p_EC = st.number_input("EC (µS/cm)", 0.0, 10000.0, 1200.0, key="cdidx_in_EC")
                p_TDS = st.number_input("TDS (mg/L)", 0.0, 5000.0, 800.0, key="cdidx_in_TDS")
                p_Cl = st.number_input("Cl- (mg/L)", 0.0, 1000.0, 45.0, key="cdidx_in_Cl")
                p_SO4 = st.number_input("SO42- (mg/L)", 0.0, 1000.0, 120.0, key="cdidx_in_SO4")
            with col2:
                p_NO3 = st.number_input("NO3- (mg/L)", 0.0, 250.0, 4.2, key="cdidx_in_NO3")
                p_Fe = st.number_input("Fe (mg/L)", 0.0, 10.0, 0.15, key="cdidx_in_Fe")
                p_Mn = st.number_input("Mn (mg/L)", 0.0, 5.0, 0.02, key="cdidx_in_Mn")
                p_Pb = st.number_input("Pb (mg/L)", 0.0, 2.0, 0.001, key="cdidx_in_Pb")
                p_Cd = st.number_input("Cd (mg/L)", 0.0, 1.0, 0.0005, key="cdidx_in_Cd")
            with col3:
                p_As = st.number_input("As (mg/L)", 0.0, 2.0, 0.002, key="cdidx_in_As")
                p_Zn = st.number_input("Zn (mg/L)", 0.0, 20.0, 1.1, key="cdidx_in_Zn")
                p_Cu = st.number_input("Cu (mg/L)", 0.0, 10.0, 0.05, key="cdidx_in_Cu")
                p_Al = st.number_input("Al (mg/L)", 0.0, 5.0, 0.08, key="cdidx_in_Al")
                p_Cr = st.number_input("Cr (mg/L)", 0.0, 2.0, 0.005, key="cdidx_in_Cr")
            with col4:
                p_Prof = st.number_input("Profundidad (m)", 0.0, 500.0, 45.0, key="cdidx_in_Prof")
                p_Long = st.number_input("Longitud (X)", -180.0, 180.0, -75.0, key="cdidx_in_Long")
                p_Lat = st.number_input("Latitud (Y)", -90.0, 90.0, -12.0, key="cdidx_in_Lat")
                p_Alt = st.number_input("Altitud (m.s.n.m.)", 0.0, 5000.0, 150.0, key="cdidx_in_Alt")
                p_Cota = st.number_input("Cota Terreno", -100.0, 5000.0, 148.0, key="cdidx_in_Cota")

            if st.button("🔮 Calcular Pronóstico de Cd Index Individual", use_container_width=True):
                vector_crudo = [[
                    p_pH, p_EC, p_TDS, p_Cl, p_SO4, p_NO3, p_Fe, p_Mn, p_Pb, p_Cd, p_As, p_Zn, p_Cu, p_Al, p_Cr,
                    p_Prof, p_Long, p_Lat, p_Alt, p_Cota
                ]]
                resultado = pronosticar_nuevos_datos_cd_index(model, scaler_X, scaler_y, vector_crudo, device)
                
                val_cd_idx = resultado[0]
                
                # Clasificación matemática estricta del Cd Index (Contamination Index)
                if val_cd_idx < 1.0:
                    estatus, color = "Bajo (< 1 - Nivel de contaminación seguro)", "green"
                elif val_cd_idx <= 3.0:
                    estatus, color = "Medio (1 a 3 - Umbral de advertencia hidroquímica)", "orange"
                else:
                    estatus, color = "Alto (> 3 - Grado de contaminación crítico por metales)", "red"

                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#f0f2f6; text-align:center;">
                    <h4>Valor del Cd Index Pronosticado por RNN: <span style="color:{color}; font-size:30px;">{val_cd_idx:.4f}</span></h4>
                    <h5>Estado de Contaminación de la Fuente: <span style="color:{color};">{estatus}</span></h5>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("##### Evaluación predictiva masiva de Cd Index por lote:")
            archivo_pronostico = st.file_uploader("Subir CSV para procesar lote Cd Index", type=["csv"], key="uploader_batch_cdidx")
            
            if archivo_pronostico is not None:
                df_nuevos_datos = pd.read_csv(archivo_pronostico)
                df_nuevos_datos.columns = df_nuevos_datos.columns.str.strip()
                
                faltan_batch = [c for c in columnas_predictores if c not in df_nuevos_datos.columns]
                
                if faltan_batch:
                    st.error(f"❌ Archivo no compatible. Faltan las columnas de la matriz: {faltan_batch}")
                else:
                    if st.button("🔮 Correr Inferencia Masiva Cd Index", use_container_width=True):
                        matrix_raw = df_nuevos_datos[columnas_predictores].values
                        pronosticos_lote = pronosticar_nuevos_datos_cd_index(model, scaler_X, scaler_y, matrix_raw, device)
                        
                        df_nuevos_datos['Cd_Index_Pronosticado_RNN'] = pronosticos_lote
                        
                        # Clasificación condicional vectorizada por NumPy para lotes masivos
                        condiciones = [
                            (df_nuevos_datos['Cd_Index_Pronosticado_RNN'] < 1.0),
                            (df_nuevos_datos['Cd_Index_Pronosticado_RNN'] >= 1.0) & (df_nuevos_datos['Cd_Index_Pronosticado_RNN'] <= 3.0),
                            (df_nuevos_datos['Cd_Index_Pronosticado_RNN'] > 3.0)
                        ]
                        opciones = ['Bajo (<1)', 'Medio (1-3)', 'Alto (>3)']
                        df_nuevos_datos['Clasificacion_Cd_Index'] = np.select(condiciones, opciones, default='Desconocido')
                        
                        st.success(f"✅ Procesamiento predictivo finalizado para {len(df_nuevos_datos)} registros.")
                        st.dataframe(df_nuevos_datos, use_container_width=True)
                        
                        csv_descarga = df_nuevos_datos.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Predicciones de Índice de Contaminación Cd Index (CSV)",
                            data=csv_descarga,
                            file_name="pronosticos_Cd_Index_AI.csv",
                            mime="text/csv",
                            use_container_width=True
                        )