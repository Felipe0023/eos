import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# DEFINICIÓN DE LA ARQUITECTURA DE LA RED DE ELMAN (Adaptada a HPI)
# =====================================================================
class ElmanRNNHPI(nn.Module):
    def __init__(self, input_size=20, hidden_size=144, output_size=1):
        super(ElmanRNNHPI, self).__init__()
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
def calcular_metricas_hpi(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    nse = r2 
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mad = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mad, nse

def pronosticar_nuevos_datos_hpi(modelo_entrenado, escalador_X, escalador_y, datos_nuevos_raw, device):
    modelo_entrenado.eval() 
    datos_array = np.array(datos_nuevos_raw, dtype=np.float32)
    if datos_array.ndim == 1:
        datos_array = datos_array.reshape(1, -1)
        
    datos_escalados = escalador_X.transform(datos_array)
    tensor_input = torch.tensor(datos_escalados, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        prediccion_tensor_scaled = modelo_entrenado(tensor_input)
        
    hpi_pronosticado = escalador_y.inverse_transform(prediccion_tensor_scaled.cpu().numpy())
    return hpi_pronosticado.flatten()

# =====================================================================
# 🌟 FUNCIÓN PRINCIPAL DEL MÓDULO PARA STREAMLIT
# =====================================================================
def MODULO_ENTRENAMIENTO_Y_PRONOSTICO_HPI():
    """
    Módulo IA homólogo diseñado exclusivamente para predecir el Índice de Contaminación 
    por Metales Pesados (HPI) utilizando 20 predictores ambientales y estructurales.
    """
    st.markdown("<h3 style='text-align: center;'>🧠 Modelado de Inteligencia Artificial para HPI (Elman RNN)</h3>", unsafe_allow_html=True)
    
    # 1. Validación de datos en session_state (debe provenir del bloque donde se calculó el HPI)
    if "K001_datos" not in st.session_state or st.session_state["K001_datos"] is None:
        st.warning("⚠️ No se detectó la matriz base con los índices calculados. Por favor, procese la pestaña de Índices de Metales Pesados primero.")
        return

    # Recuperar datos históricos
    df_historico = st.session_state["K001_datos"].copy()

    # Dispositivo de cálculo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Definición de las 20 columnas predictoras
    columnas_predictores = [
        'pH', 'EC', 'TDS', 'Cl-', 'SO42-', 'NO3-', 'Fe', 'Mn', 'Pb', 'Cd', 'As', 'Zn', 'Cu', 'Al', 'Cr',
        'Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota'
    ]
    
    # Validar consistencia de columnas y del target HPI
    columnas_faltantes = [col for col in columnas_predictores if col not in df_historico.columns]
    if columnas_faltantes:
        st.error(f"❌ Estructura incompleta en el archivo cargado. Faltan las variables: {columnas_faltantes}")
        return
    
    if 'HPI' not in df_historico.columns:
        st.error("❌ No se encontró la columna objetivo 'HPI' calculada en la matriz de datos.")
        return
        
    X_datos = df_historico[columnas_predictores].values
    y_datos = df_historico['HPI'].values.reshape(-1, 1)

    # 2. Interfaz de Control de Entrenamiento
    st.subheader("🏋️ Entrenamiento del Modelo Predictivo de Metales (HPI)")
    st.info(f"Hardware Activo: `{device.type.upper()}` | Variables de entrada de la red: `20` | Registros: `{X_datos.shape[0]}`")

    with st.expander("⚙️ Hiperparámetros de Ajuste de la Red (HPI)", expanded=False):
        lr = st.slider("Tasa de Aprendizaje (Learning Rate)", 0.001, 0.050, 0.005, step=0.001, key="lr_hpi")
        epochs = st.slider("Épocas de Entrenamiento", 50, 500, 150, step=25, key="epochs_hpi")
        hidden_size = st.slider("Unidades Ocultas (Capa de Contexto)", 32, 256, 144, step=16, key="hidden_hpi")

    if st.button("🚀 Entrenar Red Neuronal Recurrente para HPI", use_container_width=True):
        with st.spinner("Optimizando pesos de la red recurrente para detección de metales pesados..."):
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

            # Instanciación
            model = ElmanRNNHPI(input_size=20, hidden_size=hidden_size, output_size=1).to(device)
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

            # Métricas finales
            model.eval()
            with torch.no_grad():
                pred_train_scaled = model(X_train_t).cpu().numpy()
                pred_train = scaler_y.inverse_transform(pred_train_scaled).flatten()
                pred_val_scaled = model(X_val_t).cpu().numpy()
                pred_val = scaler_y.inverse_transform(pred_val_scaled).flatten()

            r2_tr, rmse_tr, mad_tr, nse_tr = calcular_metricas_hpi(y_train.flatten(), pred_train)
            r2_va, rmse_va, mad_va, nse_va = calcular_metricas_hpi(y_val.flatten(), pred_val)

            # Guardar en memoria de sesión
            st.session_state["hpi_model_fitted"] = model
            st.session_state["hpi_scaler_x"] = scaler_X
            st.session_state["hpi_scaler_y"] = scaler_y
            
            st.success("🎉 ¡Modelo RNN ajustado para predicción de HPI guardado con éxito!")

            # Cuadro estadístico de validación
            st.markdown("##### 📊 Matriz Comparativa de Validez Matemática (HPI)")
            tabla_metricas = pd.DataFrame({
                "Métrica": ["Coef. Determinación (R²)", "Raíz Error Cuadrático (RMSE)", "Desviación Absoluta Media (MAD)", "Eficiencia Nash-Sutcliffe (NSE)"],
                "Entrenamiento": [f"{r2_tr:.4f}", f"{rmse_tr:.4f}", f"{mad_tr:.4f}", f"{nse_tr:.4f}"],
                "Validación": [f"{r2_va:.4f}", f"{rmse_va:.4f}", f"{mad_va:.4f}", f"{nse_va:.4f}"]
            })
            st.dataframe(tabla_metricas, use_container_width=True, hide_index=True)

    # 3. Módulos Operativos de Pronóstico (Inferencia)
    if "hpi_model_fitted" in st.session_state:
        st.write("---")
        st.subheader("🔮 Panel Operativo de Inferencias de Toxicidad por Metales")
        
        model = st.session_state["hpi_model_fitted"]
        scaler_X = st.session_state["hpi_scaler_x"]
        scaler_y = st.session_state["hpi_scaler_y"]

        tab1, tab2 = st.tabs(["🕹️ Pronóstico Individual", "📂 Pronóstico Masivo (CSV)"])

        with tab1:
            st.markdown("##### Ingrese las variables analíticas de la nueva muestra:")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                p_pH = st.number_input("pH", 0.0, 14.0, 7.2, key="in_pH")
                p_EC = st.number_input("EC (µS/cm)", 0.0, 10000.0, 1200.0, key="in_EC")
                p_TDS = st.number_input("TDS (mg/L)", 0.0, 5000.0, 800.0, key="in_TDS")
                p_Cl = st.number_input("Cl- (mg/L)", 0.0, 1000.0, 45.0, key="in_Cl")
                p_SO4 = st.number_input("SO42- (mg/L)", 0.0, 1000.0, 120.0, key="in_SO4")
            with col2:
                p_NO3 = st.number_input("NO3- (mg/L)", 0.0, 250.0, 4.2, key="in_NO3")
                p_Fe = st.number_input("Fe (mg/L)", 0.0, 10.0, 0.15, key="in_Fe")
                p_Mn = st.number_input("Mn (mg/L)", 0.0, 5.0, 0.02, key="in_Mn")
                p_Pb = st.number_input("Pb (mg/L)", 0.0, 2.0, 0.001, key="in_Pb")
                p_Cd = st.number_input("Cd (mg/L)", 0.0, 1.0, 0.0005, key="in_Cd")
            with col3:
                p_As = st.number_input("As (mg/L)", 0.0, 2.0, 0.002, key="in_As")
                p_Zn = st.number_input("Zn (mg/L)", 0.0, 20.0, 1.1, key="in_Zn")
                p_Cu = st.number_input("Cu (mg/L)", 0.0, 10.0, 0.05, key="in_Cu")
                p_Al = st.number_input("Al (mg/L)", 0.0, 5.0, 0.08, key="in_Al")
                p_Cr = st.number_input("Cr (mg/L)", 0.0, 2.0, 0.005, key="in_Cr")
            with col4:
                p_Prof = st.number_input("Profundidad (m)", 0.0, 500.0, 45.0, key="in_Prof")
                p_Long = st.number_input("Longitud (X)", -180.0, 180.0, -75.0, key="in_Long")
                p_Lat = st.number_input("Latitud (Y)", -90.0, 90.0, -12.0, key="in_Lat")
                p_Alt = st.number_input("Altitud (m.s.n.m.)", 0.0, 5000.0, 150.0, key="in_Alt")
                p_Cota = st.number_input("Cota Terreno", -100.0, 5000.0, 148.0, key="in_Cota")

            if st.button("🔮 Calcular Pronóstico de HPI Individual", use_container_width=True):
                vector_crudo = [[
                    p_pH, p_EC, p_TDS, p_Cl, p_SO4, p_NO3, p_Fe, p_Mn, p_Pb, p_Cd, p_As, p_Zn, p_Cu, p_Al, p_Cr,
                    p_Prof, p_Long, p_Lat, p_Alt, p_Cota
                ]]
                resultado = pronosticar_nuevos_datos_hpi(model, scaler_X, scaler_y, vector_crudo, device)
                
                val_hpi = resultado[0]
                # Clasificación oficial del HPI
                if val_hpi < 100:
                    estatus, color = "Bajo (Seguro - Dentro de límites admisibles)", "green"
                else:
                    estatus, color = "Crítico (Peligro - Presencia severa de metales pesados)", "red"

                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#f0f2f6; text-align:center;">
                    <h4>Valor HPI Pronosticado por Red de Elman: <span style="color:{color}; font-size:30px;">{val_hpi:.2f}</span></h4>
                    <h5>Condición de la Fuente: <span style="color:{color};">{estatus}</span></h5>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("##### Evaluación predictiva masiva por lote:")
            archivo_pronostico = st.file_uploader("Subir CSV para procesar lote HPI", type=["csv"], key="uploader_batch_hpi")
            
            if archivo_pronostico is not None:
                df_nuevos_datos = pd.read_csv(archivo_pronostico)
                df_nuevos_datos.columns = df_nuevos_datos.columns.str.strip()
                
                faltan_batch = [c for c in columnas_predictores if c not in df_nuevos_datos.columns]
                
                if faltan_batch:
                    st.error(f"❌ Archivo no compatible. Faltan las columnas de la matriz: {faltan_batch}")
                else:
                    if st.button("🔮 Correr Inferencia Masiva HPI", use_container_width=True):
                        matrix_raw = df_nuevos_datos[columnas_predictores].values
                        pronosticos_lote = pronosticar_nuevos_datos_hpi(model, scaler_X, scaler_y, matrix_raw, device)
                        
                        df_nuevos_datos['HPI_Pronosticado_RNN'] = pronosticos_lote
                        df_nuevos_datos['Clase_HPI_Estimada'] = np.where(df_nuevos_datos['HPI_Pronosticado_RNN'] < 100, "Bajo (<100)", "Crítico (≥100)")
                        
                        st.success(f"✅ Procesamiento predictivo finalizado para {len(df_nuevos_datos)} puntos de muestreo.")
                        st.dataframe(df_nuevos_datos, use_container_width=True)
                        
                        csv_descarga = df_nuevos_datos.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Predicciones de Metales Pesados HPI (CSV)",
                            data=csv_descarga,
                            file_name="pronosticos_HPI_AI.csv",
                            mime="text/csv",
                            use_container_width=True
                        )