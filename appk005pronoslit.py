import streamlit as st
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# DEFINICIÓN DE LA ARQUITECTURA DE LA RED DE ELMAN (Adaptada a 20 Inputs)
# =====================================================================
class ElmanRNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=144, output_size=1):
        super(ElmanRNN, self).__init__()
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
def calcular_metricas(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    nse = r2 
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mad = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mad, nse

def pronosticar_nuevos_datos(modelo_entrenado, escalador_X, escalador_y, datos_nuevos_raw, device):
    modelo_entrenado.eval() 
    datos_array = np.array(datos_nuevos_raw, dtype=np.float32)
    if datos_array.ndim == 1:
        datos_array = datos_array.reshape(1, -1)
        
    datos_escalados = escalador_X.transform(datos_array)
    tensor_input = torch.tensor(datos_escalados, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        prediccion_tensor_scaled = modelo_entrenado(tensor_input)
        
    dwqi_pronosticado = escalador_y.inverse_transform(prediccion_tensor_scaled.cpu().numpy())
    return dwqi_pronosticado.flatten()

# =====================================================================
# 🌟 FUNCIÓN PRINCIPAL DEL MÓDULO PARA STREAMLIT
# =====================================================================
def MODULO_ENTRENAMIENTO_Y_PRONOSTICO_DWQI():
    """
    Función contenedora que ejecuta el entrenamiento de la Elman RNN utilizando 
    las 15 variables químicas y 5 físicas (20 predictores totales) basándose en 
    st.session_state['datos_procesados_DWQI'].
    """
    st.markdown("<h3 style='text-align: center;'>🧠 Modelado de Inteligencia Artificial (Elman RNN)</h3>", unsafe_allow_html=True)
    
    # 1. Validación de dependencias y datos de origen upstream
    if "datos_procesados_DWQI" not in st.session_state or st.session_state["datos_procesados_DWQI"] is None:
        st.warning("⚠️ No se han encontrado datos analizados previamente. Por favor, calcule el DWQI en el módulo correspondiente antes de proceder al entrenamiento.")
        return

    # Recuperar datos históricos procesados
    df_historico = st.session_state["datos_procesados_DWQI"].copy()

    # Asignación dinámica de hardware (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 💡 NUEVA CONDICIÓN: Matriz extendida a 20 características (15 Hidroquímicas + 5 Físico-Estructurales)
    columnas_predictores = [
        'pH', 'EC', 'TDS', 'Cl-', 'SO42-', 'NO3-', 'Fe', 'Mn', 'Pb', 'Cd', 'As', 'Zn', 'Cu', 'Al', 'Cr',
        'Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota'
    ]
    
    # Validar que todas las columnas requeridas por el modelo existan en el DataFrame del Session State
    columnas_faltantes = [col for col in columnas_predictores if col not in df_historico.columns]
    if columnas_faltantes:
        st.error(f"❌ Estructura incompleta. Faltan las siguientes columnas esenciales para el entrenamiento: {columnas_faltantes}")
        return
        
    X_datos = df_historico[columnas_predictores].values
    y_datos = df_historico['DWQI'].values.reshape(-1, 1)

    # 2. Interfaz de Entrenamiento de la Red
    st.subheader("🏋️ Entrenamiento del Modelo Predictivo")
    st.info(f"Dispositivo activo: `{device.type.upper()}` | Características de entrada: `20` | Muestras históricas: `{X_datos.shape[0]}`")

    with st.expander("⚙️ Hiperparámetros de la Elman RNN", expanded=False):
        lr = st.slider("Tasa de Aprendizaje (Learning Rate)", 0.001, 0.050, 0.005, step=0.001)
        epochs = st.slider("Épocas de Entrenamiento", 50, 500, 150, step=25)
        hidden_size = st.slider("Unidades Ocultas (Capa de Contexto)", 32, 256, 144, step=16)

    # Botón para detonar el entrenamiento
    if st.button("🚀 Entrenar Red Neuronal Recurrente", use_container_width=True):
        with st.spinner("Procesando datos y ajustando pesos de la red de 20 entradas..."):
            # Fijar semillas para reproducibilidad controlada en Streamlit
            np.random.seed(42)
            torch.manual_seed(42)

            # Separación y Escalado de Datos
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

            # Inicializar componentes e instanciar arquitectura de Elman con input_size=20
            model = ElmanRNN(input_size=20, hidden_size=hidden_size, output_size=1).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Contenedor para visualización dinámica del progreso
            progreso_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss_train = criterion(outputs, y_train_t)
                loss_train.backward()
                optimizer.step()

                # Actualización de la barra de progreso
                progreso_bar.progress((epoch + 1) / epochs)
                if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_t)
                        loss_val = criterion(val_outputs, y_val_t)
                    status_text.text(f"Época [{epoch+1}/{epochs}] -> Loss Train: {loss_train.item():.4f} | Loss Val: {loss_val.item():.4f}")

            # Evaluación Final de Validez Matemática
            model.eval()
            with torch.no_grad():
                pred_train_scaled = model(X_train_t).cpu().numpy()
                pred_train = scaler_y.inverse_transform(pred_train_scaled).flatten()
                pred_val_scaled = model(X_val_t).cpu().numpy()
                pred_val = scaler_y.inverse_transform(pred_val_scaled).flatten()

            r2_tr, rmse_tr, mad_tr, nse_tr = calcular_metricas(y_train.flatten(), pred_train)
            r2_va, rmse_va, mad_va, nse_va = calcular_metricas(y_val.flatten(), pred_val)

            # Almacenar artefactos entrenados en session_state para que persistan en la app
            st.session_state["rnn_model_fitted"] = model
            st.session_state["rnn_scaler_x"] = scaler_X
            st.session_state["rnn_scaler_y"] = scaler_y
            
            st.success("🎉 ¡Modelo adaptado de 20 variables entrenado con éxito y guardado!")

            # Despliegue de la Matriz Comparativa de Validez en Streamlit
            st.markdown("##### 📊 Matriz Comparativa de Validez")
            tabla_metricas = pd.DataFrame({
                "Métrica": ["Coef. Determinación (R²)", "Raíz Error Cuadrático (RMSE)", "Desviación Absoluta Media (MAD)", "Eficiencia Nash-Sutcliffe (NSE)"],
                "Entrenamiento": [f"{r2_tr:.4f}", f"{rmse_tr:.4f}", f"{mad_tr:.4f}", f"{nse_tr:.4f}"],
                "Validación": [f"{r2_va:.4f}", f"{rmse_va:.4f}", f"{mad_va:.4f}", f"{nse_va:.4f}"]
            })
            st.dataframe(tabla_metricas, use_container_width=True, hide_index=True)

    # 3. Operaciones de Inferencia y Pronóstico con los Modelos Guardados
    if "rnn_model_fitted" in st.session_state:
        st.write("---")
        st.subheader("🔮 Artefactos Operativos de Pronóstico")
        
        # Recuperar artefactos entrenados desde el st.session_state
        model = st.session_state["rnn_model_fitted"]
        scaler_X = st.session_state["rnn_scaler_x"]
        scaler_y = st.session_state["rnn_scaler_y"]

        tab1, tab2 = st.tabs(["🕹️ Pronóstico Individual por Muestra", "📂 Pronóstico Masivo mediante Archivo"])

        # ARTEFACTO 1: Inferencia Manual e Individual (Adaptada a las 20 variables)
        with tab1:
            st.markdown("##### Ingrese los parámetros fisicoquímicos y geográficos:")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                p_pH = st.number_input("pH", 0.0, 14.0, 7.2)
                p_EC = st.number_input("EC (µS/cm)", 0.0, 10000.0, 1200.0)
                p_TDS = st.number_input("TDS (mg/L)", 0.0, 5000.0, 800.0)
                p_Cl = st.number_input("Cl- (mg/L)", 0.0, 1000.0, 45.0)
                p_SO4 = st.number_input("SO42- (mg/L)", 0.0, 1000.0, 120.0)
            with col2:
                p_NO3 = st.number_input("NO3- (mg/L)", 0.0, 250.0, 4.2)
                p_Fe = st.number_input("Fe (mg/L)", 0.0, 10.0, 0.15)
                p_Mn = st.number_input("Mn (mg/L)", 0.0, 5.0, 0.02)
                p_Pb = st.number_input("Pb (mg/L)", 0.0, 2.0, 0.001)
                p_Cd = st.number_input("Cd (mg/L)", 0.0, 1.0, 0.0005)
            with col3:
                p_As = st.number_input("As (mg/L)", 0.0, 2.0, 0.002)
                p_Zn = st.number_input("Zn (mg/L)", 0.0, 20.0, 1.1)
                p_Cu = st.number_input("Cu (mg/L)", 0.0, 10.0, 0.05)
                p_Al = st.number_input("Al (mg/L)", 0.0, 5.0, 0.08)
                p_Cr = st.number_input("Cr (mg/L)", 0.0, 2.0, 0.005)
            with col4:
                # 💡 VARIABLES NUEVAS INYECTADAS A LA INTERFAZ MANUAL
                p_Prof = st.number_input("Profundidad (m)", 0.0, 500.0, 45.0)
                p_Long = st.number_input("Longitud (X)", -180.0, 180.0, -75.0)
                p_Lat = st.number_input("Latitud (Y)", -90.0, 90.0, -12.0)
                p_Alt = st.number_input("Altitud (m.s.n.m.)", 0.0, 5000.0, 150.0)
                p_Cota = st.number_input("Cota Terreno", -100.0, 5000.0, 148.0)

            if st.button("🔮 Calcular Pronóstico de DWQI Individual", use_container_width=True):
                # Construcción del vector respetando el orden estricto de columnas_predictores
                vector_crudo = [[
                    p_pH, p_EC, p_TDS, p_Cl, p_SO4, p_NO3, p_Fe, p_Mn, p_Pb, p_Cd, p_As, p_Zn, p_Cu, p_Al, p_Cr,
                    p_Prof, p_Long, p_Lat, p_Alt, p_Cota
                ]]
                resultado = pronosticar_nuevos_datos(model, scaler_X, scaler_y, vector_crudo, device)
                
                val_dwqi = resultado[0]
                if val_dwqi <= 25: estatus, color = "Excelente", "green"
                elif val_dwqi <= 50: estatus, color = "Buena", "blue"
                elif val_dwqi <= 75: estatus, color = "Pobre (Regular)", "orange"
                elif val_dwqi <= 100: estatus, color = "Muy Pobre", "red"
                else: estatus, color = "No apta para consumo (Inadecuada)", "purple"

                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#f0f2f6; text-align:center;">
                    <h4>Valor del Índice DWQI Pronosticado por RNN: <span style="color:{color}; font-size:30px;">{val_dwqi:.2f}</span></h4>
                    <h5>Clasificación Estimada: <span style="color:{color};">{estatus}</span></h5>
                </div>
                """, unsafe_allow_html=True)

        # ARTEFACTO 2: Inferencia Batch (Lotes Masivos por CSV)
        with tab2:
            st.markdown("##### Suba un nuevo archivo CSV para pronosticar múltiples pozos simultáneamente:")
            st.caption("El archivo cargado debe contener obligatoriamente los 15 parámetros químicos y las 5 variables físicas solicitadas.")
            
            archivo_pronostico = st.file_uploader("Subir CSV de Nuevos Datos", type=["csv"], key="uploader_batch_rnn")
            
            if archivo_pronostico is not None:
                df_nuevos_datos = pd.read_csv(archivo_pronostico)
                df_nuevos_datos.columns = df_nuevos_datos.columns.str.strip()
                
                # Validar la presencia de las 20 columnas antes de alimentar a la traza de evaluación de PyTorch
                faltan_batch = [c for c in columnas_predictores if c not in df_nuevos_datos.columns]
                
                if faltan_batch:
                    st.error(f"❌ Estructura de archivo incorrecta. El modelo de 20 variables no encuentra: {faltan_batch}")
                else:
                    if st.button("🔮 Procesar Pronóstico Masivo de Lote", use_container_width=True):

                        # ---------------------------------------------------------
                        # [OPCIÓN A INTEGRADA] - IMPUTACIÓN POR LA MEDIA PARA PRONÓSTICO
                        # ---------------------------------------------------------
                        # Detectar si hay valores nulos en las columnas predictoras
                        if df_nuevos_datos[columnas_predictores].isnull().sum().sum() > 0:
                            # Rellenar cada valor NaN con la media de su respectiva columna
                            df_nuevos_datos[columnas_predictores] = df_nuevos_datos[columnas_predictores].fillna(df_nuevos_datos[columnas_predictores].mean())
                            st.info("ℹ️ Se detectaron valores nulos en el archivo CSV. Se aplicó imputación automática por la media del lote para poder calcular los pronósticos.")
                        # ---------------------------------------------------------




                        
                        matrix_raw = df_nuevos_datos[columnas_predictores].values
                        pronosticos_lote = pronosticar_nuevos_datos(model, scaler_X, scaler_y, matrix_raw, device)
                        
                        # Guardar predicciones en el DataFrame y mostrar resultados
                        df_nuevos_datos['DWQI_Pronosticado_RNN'] = pronosticos_lote
                        
                        st.success(f"✅ Procesamiento completado. Se generaron {len(df_nuevos_datos)} pronósticos con la matriz expandida.")
                        st.dataframe(df_nuevos_datos, use_container_width=True)
                        
                        # Habilitar descarga inmediata de los datos procesados por la IA
                        csv_descarga = df_nuevos_datos.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Respuestas del Pronóstico (CSV)",
                            data=csv_descarga,
                            file_name="pronosticos_DWQI_AI_20var.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


