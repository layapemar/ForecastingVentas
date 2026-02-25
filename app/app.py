import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #667eea !important;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #764ba2;
    }
    .highlight-bf {
        background-color: #fff3cd;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos y modelo
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = joblib.load('models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia ya transformado"""
    try:
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# Función para obtener columnas del modelo
def obtener_columnas_modelo(modelo):
    """Extrae las columnas que espera el modelo"""
    if hasattr(modelo, 'feature_names_in_'):
        return modelo.feature_names_in_.tolist()
    return None

# Función para realizar predicciones recursivas
def predecir_recursivo(df_producto, modelo, columnas_modelo):
    """
    Realiza predicciones día por día actualizando los lags recursivamente
    """
    df_pred = df_producto.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    predicciones = []
    
    for idx in range(len(df_pred)):
        # Preparar features para predicción
        X = df_pred.loc[[idx], columnas_modelo]
        
        # Predecir
        pred = modelo.predict(X)[0]
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if idx < len(df_pred) - 1:
            # Desplazar lags
            df_pred.loc[idx + 1, 'lag_7'] = df_pred.loc[idx, 'lag_6']
            df_pred.loc[idx + 1, 'lag_6'] = df_pred.loc[idx, 'lag_5']
            df_pred.loc[idx + 1, 'lag_5'] = df_pred.loc[idx, 'lag_4']
            df_pred.loc[idx + 1, 'lag_4'] = df_pred.loc[idx, 'lag_3']
            df_pred.loc[idx + 1, 'lag_3'] = df_pred.loc[idx, 'lag_2']
            df_pred.loc[idx + 1, 'lag_2'] = df_pred.loc[idx, 'lag_1']
            df_pred.loc[idx + 1, 'lag_1'] = pred
            
            # Actualizar media móvil
            ultimas_7 = predicciones[-7:] if len(predicciones) >= 7 else predicciones
            df_pred.loc[idx + 1, 'media_movil_7d'] = np.mean(ultimas_7)
    
    return predicciones

# Función para aplicar ajustes de simulación
def aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia):
    """
    Aplica los ajustes de descuento y competencia al dataframe
    """
    df_ajustado = df_producto.copy()
    
    # Ajustar descuento y precio_venta
    factor_descuento = 1 + (ajuste_descuento / 100)
    descuento_actual = (df_ajustado['precio_base'] - df_ajustado['precio_venta']) / df_ajustado['precio_base']
    nuevo_descuento = descuento_actual * factor_descuento
    nuevo_descuento = nuevo_descuento.clip(0, 0.5)  # Limitar entre 0% y 50%
    
    df_ajustado['precio_venta'] = df_ajustado['precio_base'] * (1 - nuevo_descuento)
    df_ajustado['porcentaje_descuento'] = nuevo_descuento * 100
    
    # Ajustar precios de competencia
    if escenario_competencia == "Competencia -5%":
        factor_comp = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_comp = 1.05
    else:
        factor_comp = 1.0
    
    # Ajustar precios individuales de competidores si existen
    competidores = ['Amazon', 'Decathlon', 'Deporvillage']
    for comp in competidores:
        if comp in df_ajustado.columns:
            df_ajustado[comp] = df_ajustado[comp] * factor_comp
    
    # Recalcular precio_competencia
    df_ajustado['precio_competencia'] = df_ajustado['precio_competencia'] * factor_comp
    
    # Recalcular ratio_precio
    df_ajustado['ratio_precio'] = df_ajustado['precio_venta'] / df_ajustado['precio_competencia']
    
    return df_ajustado

# Cargar modelo y datos
modelo = cargar_modelo()
df_inferencia = cargar_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# Obtener columnas del modelo
columnas_modelo = obtener_columnas_modelo(modelo)

# Inicializar session_state
if 'simulacion_ejecutada' not in st.session_state:
    st.session_state.simulacion_ejecutada = False
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

# ====================
# SIDEBAR - CONTROLES
# ====================
st.sidebar.markdown("# 🎛️ Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
productos = sorted(df_inferencia['nombre'].unique().tolist())
producto_seleccionado = st.sidebar.selectbox(
    "📦 Selecciona un producto:",
    productos,
    help="Elige el producto que deseas simular"
)

# Slider de descuento
st.sidebar.markdown("### 💰 Ajuste de Descuento")
ajuste_descuento = st.sidebar.slider(
    "Modificar descuento actual:",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Ajusta el descuento actual. Ej: +20% aumenta el descuento actual en un 20%"
)

# Escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "Selecciona el escenario:",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    help="Simula cambios en los precios de la competencia"
)

# Botón de simulación
st.sidebar.markdown("---")
simular = st.sidebar.button("🚀 Simular Ventas", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Ajusta los controles y presiona 'Simular Ventas' para ver las proyecciones.")

# ====================
# ZONA PRINCIPAL
# ====================

# Header
st.markdown(f"# 📊 Dashboard de Simulación - Noviembre 2025")
st.markdown(f"### Producto: **{producto_seleccionado}**")
st.markdown("---")

# Ejecutar simulación cuando se presiona el botón
if simular:
    with st.spinner("🔄 Procesando predicciones recursivas..."):
        # Filtrar datos del producto
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        # Aplicar ajustes
        df_ajustado = aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia)
        
        # Realizar predicciones recursivas
        predicciones = predecir_recursivo(df_ajustado, modelo, columnas_modelo)
        
        # Agregar predicciones al dataframe
        df_ajustado = df_ajustado.sort_values('fecha').reset_index(drop=True)
        df_ajustado['unidades_predichas'] = predicciones
        df_ajustado['ingresos_proyectados'] = df_ajustado['unidades_predichas'] * df_ajustado['precio_venta']
        
        # Calcular comparativa de escenarios
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        for escenario in escenarios:
            df_esc = aplicar_ajustes(df_producto, ajuste_descuento, escenario)
            preds_esc = predecir_recursivo(df_esc, modelo, columnas_modelo)
            df_esc_sorted = df_esc.sort_values('fecha').reset_index(drop=True)
            df_esc_sorted['unidades_predichas'] = preds_esc
            df_esc_sorted['ingresos_proyectados'] = df_esc_sorted['unidades_predichas'] * df_esc_sorted['precio_venta']
            
            resultados_escenarios[escenario] = {
                'unidades': df_esc_sorted['unidades_predichas'].sum(),
                'ingresos': df_esc_sorted['ingresos_proyectados'].sum()
            }
        
        # Guardar en session_state
        st.session_state.resultados = {
            'df_ajustado': df_ajustado,
            'producto': producto_seleccionado,
            'ajuste_descuento': ajuste_descuento,
            'escenario_competencia': escenario_competencia,
            'resultados_escenarios': resultados_escenarios
        }
        st.session_state.simulacion_ejecutada = True

# Mostrar resultados si existe una simulación
if st.session_state.simulacion_ejecutada and st.session_state.resultados is not None:
    resultados = st.session_state.resultados
    df_ajustado = resultados['df_ajustado']
    resultados_escenarios = resultados['resultados_escenarios']
    
    # Calcular KPIs
    unidades_totales = df_ajustado['unidades_predichas'].sum()
    ingresos_totales = df_ajustado['ingresos_proyectados'].sum()
    precio_promedio = df_ajustado['precio_venta'].mean()
    descuento_promedio = df_ajustado['porcentaje_descuento'].mean()
    
    # Información de los parámetros usados
    st.info(f"📋 **Última simulación:** Descuento {resultados['ajuste_descuento']:+d}% | Escenario: {resultados['escenario_competencia']}")
        
# Información de los parámetros usados
    st.info(f"📋 **Última simulación:** Descuento {resultados['ajuste_descuento']:+d}% | Escenario: {resultados['escenario_competencia']}")
        
    # ====================
    # KPIs DESTACADOS
    # ====================
    st.markdown("## 📈 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🛒 Unidades Totales",
            value=f"{int(unidades_totales):,}",
            help="Total de unidades proyectadas para noviembre"
        )
    
    with col2:
        st.metric(
            label="💰 Ingresos Proyectados",
            value=f"{ingresos_totales:,.2f} €",
            help="Ingresos totales estimados"
        )
    
    with col3:
        st.metric(
            label="💵 Precio Promedio",
            value=f"{precio_promedio:.2f} €",
            help="Precio de venta promedio"
        )
    
    with col4:
        st.metric(
            label="🏷️ Descuento Promedio",
            value=f"{descuento_promedio:.1f}%",
            help="Descuento medio aplicado"
        )
    
    st.markdown("---")
    st.markdown("---")
        
    # ====================
    # GRÁFICO DE PREDICCIÓN DIARIA
    # ====================
    st.markdown("## 📅 Predicción Diaria de Ventas")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Estilo seaborn
    sns.set_style("whitegrid")
    
    # Gráfico de línea
    dias = df_ajustado['dia_mes'].values
    ventas = df_ajustado['unidades_predichas'].values
    
    ax.plot(dias, ventas, marker='o', linewidth=2.5, markersize=6, 
            color='#667eea', label='Unidades Predichas')
    
    # Marcar Black Friday (día 28)
    idx_bf = df_ajustado[df_ajustado['dia_mes'] == 28].index[0]
    dia_bf = df_ajustado.loc[idx_bf, 'dia_mes']
    ventas_bf = df_ajustado.loc[idx_bf, 'unidades_predichas']
    
    ax.axvline(x=dia_bf, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(dia_bf, ventas_bf, 'ro', markersize=12, label='Black Friday')
    ax.annotate('🛍️ Black Friday', 
               xy=(dia_bf, ventas_bf), 
               xytext=(dia_bf-3, ventas_bf*1.15),
               fontsize=11, color='red', weight='bold',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Día de Noviembre', fontsize=12, weight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, weight='bold')
    ax.set_title('Proyección de Ventas - Noviembre 2025', fontsize=14, weight='bold', color='#764ba2')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 31))
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.markdown("---")
        
    # ====================
    # TABLA DETALLADA
    # ====================
    st.markdown("## 📋 Detalle Diario")
    
    # Preparar tabla
    df_tabla = df_ajustado[['fecha', 'dia_mes', 'precio_venta', 'precio_competencia', 
                             'porcentaje_descuento', 'unidades_predichas', 'ingresos_proyectados']].copy()
    
    # Añadir día de la semana
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_tabla['dia_semana'] = df_tabla['fecha'].dt.dayofweek.map(lambda x: dias_semana[x])
    
    # Marcar Black Friday
    df_tabla['Black Friday'] = df_tabla['dia_mes'].apply(lambda x: '🛍️' if x == 28 else '')
    
    # Formatear columnas
    df_tabla['Fecha'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
    df_tabla['Día'] = df_tabla['dia_semana']
    df_tabla['Precio Venta'] = df_tabla['precio_venta'].apply(lambda x: f"{x:.2f} €")
    df_tabla['Precio Competencia'] = df_tabla['precio_competencia'].apply(lambda x: f"{x:.2f} €")
    df_tabla['Descuento'] = df_tabla['porcentaje_descuento'].apply(lambda x: f"{x:.1f}%")
    df_tabla['Unidades'] = df_tabla['unidades_predichas'].apply(lambda x: f"{int(x):,}")
    df_tabla['Ingresos'] = df_tabla['ingresos_proyectados'].apply(lambda x: f"{x:,.2f} €")
    
    # Seleccionar columnas finales
    df_display = df_tabla[['Black Friday', 'Fecha', 'Día', 'Precio Venta', 'Precio Competencia', 
                            'Descuento', 'Unidades', 'Ingresos']]
    
    # Mostrar tabla con estilo
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("---")
        
    # ====================
    # COMPARATIVA DE ESCENARIOS
    # ====================
    st.markdown("## 🔄 Comparativa de Escenarios de Competencia")
    st.markdown("*Comparación manteniendo el descuento seleccionado y variando solo los precios de competencia*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Actual (0%)")
        st.metric(
            label="Unidades Totales",
            value=f"{int(resultados_escenarios['Actual (0%)']['unidades']):,}"
        )
        st.metric(
            label="Ingresos Totales",
            value=f"{resultados_escenarios['Actual (0%)']['ingresos']:,.2f} €"
        )
    
    with col2:
        st.markdown("### 📉 Competencia -5%")
        diff_unidades_1 = resultados_escenarios["Competencia -5%"]['unidades'] - resultados_escenarios['Actual (0%)']['unidades']
        diff_ingresos_1 = resultados_escenarios["Competencia -5%"]['ingresos'] - resultados_escenarios['Actual (0%)']['ingresos']
        
        st.metric(
            label="Unidades Totales",
            value=f"{int(resultados_escenarios['Competencia -5%']['unidades']):,}",
            delta=f"{int(diff_unidades_1):+,}"
        )
        st.metric(
            label="Ingresos Totales",
            value=f"{resultados_escenarios['Competencia -5%']['ingresos']:,.2f} €",
            delta=f"{diff_ingresos_1:+,.2f} €"
        )
    
    with col3:
        st.markdown("### 📈 Competencia +5%")
        diff_unidades_2 = resultados_escenarios["Competencia +5%"]['unidades'] - resultados_escenarios['Actual (0%)']['unidades']
        diff_ingresos_2 = resultados_escenarios["Competencia +5%"]['ingresos'] - resultados_escenarios['Actual (0%)']['ingresos']
        
        st.metric(
            label="Unidades Totales",
            value=f"{int(resultados_escenarios['Competencia +5%']['unidades']):,}",
            delta=f"{int(diff_unidades_2):+,}"
        )
        st.metric(
            label="Ingresos Totales",
            value=f"{resultados_escenarios['Competencia +5%']['ingresos']:,.2f} €",
            delta=f"{diff_ingresos_2:+,.2f} €"
        )
    
    st.success("✅ Simulación completada con éxito")

else:
    # Mostrar instrucciones iniciales
    st.info("👈 Configura los parámetros en el panel lateral y presiona **'Simular Ventas'** para comenzar.")
    
    # Mostrar información del producto seleccionado
    df_producto_info = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Información del Producto")
        st.write(f"**Categoría:** {df_producto_info['categoria']}")
        st.write(f"**Subcategoría:** {df_producto_info['subcategoria']}")
        st.write(f"**Precio Base:** {df_producto_info['precio_base']:.2f} €")
        st.write(f"**Producto Estrella:** {'✅ Sí' if df_producto_info['es_estrella'] else '❌ No'}")
    
    with col2:
        st.markdown("### 🎯 Próximos Pasos")
        st.write("1️⃣ Selecciona un producto")
        st.write("2️⃣ Ajusta el descuento si deseas")
        st.write("3️⃣ Elige un escenario de competencia")
        st.write("4️⃣ Presiona 'Simular Ventas'")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #667eea;'>"
    "📊 Dashboard de Simulación de Ventas | Noviembre 2025 | Powered by ML"
    "</div>",
    unsafe_allow_html=True
)