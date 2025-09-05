import streamlit as st
import pandas as pd
import re
from io import BytesIO
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Subjuntivo Español con NLTK",
    page_icon="📝",
    layout="wide"
)

# Título y descripción
st.title("🔍 Analizador de Modo Subjuntivo en Español con NLTK")
st.markdown("""
Esta aplicación utiliza procesamiento de lenguaje natural (NLTK) para identificar y analizar 
todas las formas verbales en modo subjuntivo en textos en español.
""")

# Función para descargar recursos de NLTK con manejo de errores
def descargar_recursos_nltk():
    recursos_necesarios = ['punkt', 'punkt_tab']
    
    for recurso in recursos_necesarios:
        try:
            nltk.data.find(f'tokenizers/{recurso}')
        except LookupError:
            try:
                st.info(f"📥 Descargando recurso de NLTK: {recurso}")
                nltk.download(recurso, quiet=True)
                st.success(f"✅ Recurso {recurso} descargado correctamente")
            except Exception as e:
                st.warning(f"⚠️ No se pudo descargar {recurso}: {str(e)}")
                return False
    return True

# Descargar recursos necesarios
if descargar_recursos_nltk():
    st.session_state.nltk_disponible = True
else:
    st.session_state.nltk_disponible = False
    st.warning("""
    ⚠️ Algunos recursos de NLTK no están disponibles. 
    La aplicación funcionará con un método alternativo para el análisis.
    """)

# Inicializar el stemmer en español
try:
    stemmer = SnowballStemmer("spanish")
    st.session_state.stemmer_disponible = True
except:
    st.session_state.stemmer_disponible = False
    st.warning("El stemmer para español no está disponible.")

# Verbos irregulares comunes en subjuntivo
verbos_irregulares_subjuntivo = [
    'sea', 'seas', 'seamos', 'sean',  # ser
    'vaya', 'vayas', 'vayamos', 'vayan',  # ir
    'haya', 'hayas', 'hayamos', 'hayan',  # haber
    'esté', 'estés', 'estemos', 'estén',  # estar
    'dé', 'des', 'demos', 'den',  # dar
    'sepa', 'sepas', 'sepamos', 'sepan',  # saber
    'quepa', 'quepas', 'quepamos', 'quepan',  # caber
    'haga', 'hagas', 'hagamos', 'hagan',  # hacer
    'pueda', 'puedas', 'podamos', 'puedan',  # poder
    'quiera', 'quieras', 'queramos', 'quieran',  # querer
    'tenga', 'tengas', 'tengamos', 'tengan',  # tener
    'venga', 'vengas', 'vengamos', 'vengan',  # venir
    'digas', 'diga', 'digamos', 'digan',  # decir
    'oyas', 'oiga', 'oigamos', 'oigan'  # oír
]

# Conectores que suelen introducir subjuntivo
conectores_subjuntivo = [
    'que', 'cuando', 'si', 'aunque', 'para que', 'a fin de que', 
    'como si', 'a menos que', 'con tal de que', 'en caso de que',
    'sin que', 'antes de que', 'ojalá', 'espero que', 'dudo que',
    'no creo que', 'es posible que', 'es probable que', 'quizás',
    'tal vez', 'a no ser que', 'salvo que', 'excepto que'
]

# Lista de terminaciones de verbos en subjuntivo
subjuntivo_terminaciones = [
    'ara', 'aras', 'áramos', 'aran',  # Pretérito imperfecto (-ar)
    'are', 'ares', 'áremos', 'aren',  # Futuro simple (-ar)
    'iera', 'ieras', 'iéramos', 'ieran',  # Pretérito imperfecto (-er/-ir)
    'iere', 'ieres', 'iéremos', 'ieren',  # Futuro simple (-er/-ir)
    'era', 'eras', 'éramos', 'eran',  # Variante (-er)
    'ese', 'eses', 'ésemos', 'esen',  # Pretérito imperfecto (variante)
    'a', 'as', 'amos', 'an',  # Presente (-ar)
    'e', 'es', 'emos', 'en',  # Presente (-er)
    'a', 'as', 'amos', 'an',  # Presente (-ir)
    'se', 'ses', 'semos', 'sen'  # Otra variante
]

def tokenizar_texto(texto):
    """Tokeniza el texto usando NLTK o un método alternativo si NLTK no está disponible"""
    if st.session_state.nltk_disponible:
        try:
            return word_tokenize(texto, language='spanish')
        except:
            # Fallback si hay error con NLTK
            st.session_state.nltk_disponible = False
            return tokenizar_manual(texto)
    else:
        return tokenizar_manual(texto)

def tokenizar_manual(texto):
    """Tokenización manual para cuando NLTK no está disponible"""
    # Expresión regular para tokenizar palabras y signos de puntuación
    tokens = re.findall(r'\w+|[^\w\s]', texto, re.UNICODE)
    return tokens

def analizar_con_nltk(texto):
    """Analiza el texto para identificar verbos en subjuntivo"""
    # Tokenizar el texto
    tokens = tokenizar_texto(texto)
    
    resultados = []
    
    for i, palabra in enumerate(tokens):
        if es_verbo_subjuntivo(palabra):
            # Encontrar la cláusula
            clausula = encontrar_clausula_subjuntivo(tokens, i)
            
            # Determinar tiempo verbal aproximado
            tiempo = determinar_tiempo_verbal(palabra)
            
            # Determinar persona y número
            persona = determinar_persona(palabra)
            
            # Obtener lema (forma infinitiva)
            lema = obtener_lema_verbal(palabra)
            
            resultados.append({
                'Verbo': palabra,
                'Lema': lema,
                'Tiempo': tiempo,
                'Persona': persona,
                'Cláusula': clausula,
                'Posición': f"Token {i+1}"
            })
    
    return resultados

def es_verbo_subjuntivo(palabra):
    """Determina si una palabra es un verbo en subjuntivo"""
    # Limpiar la palabra de signos de puntuación
    palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())
    
    if not palabra_limpia:
        return False
    
    # Verificar verbos irregulares
    if palabra_limpia in verbos_irregulares_subjuntivo:
        return True
    
    # Verificar por terminaciones típicas del subjuntivo
    for terminacion in subjuntivo_terminaciones:
        if palabra_limpia.endswith(terminacion):
            return True
    
    return False

def encontrar_clausula_subjuntivo(tokens, posicion_verbo):
    """Encuentra la cláusula que contiene el verbo en subjuntivo"""
    # Buscar hacia atrás para encontrar el inicio de la cláusula
    inicio = max(0, posicion_verbo - 10)  # Límite para no buscar demasiado atrás
    
    for i in range(posicion_verbo, max(0, posicion_verbo - 15), -1):
        if i < len(tokens) and tokens[i].lower() in conectores_subjuntivo:
            inicio = i
            break
    
    # Buscar hacia adelante para encontrar el final de la cláusula
    fin = min(len(tokens), posicion_verbo + 15)  # Límite para no buscar demasiado adelante
    
    for i in range(posicion_verbo, min(len(tokens), posicion_verbo + 20)):
        if i < len(tokens) and tokens[i] in ['.', '!', '?', ';']:
            fin = i + 1
            break
    
    # Construir la cláusula
    clausula = ' '.join(tokens[inicio:fin])
    return clausula

def determinar_tiempo_verbal(verbo):
    """Determina el tiempo verbal aproximado basado en la terminación"""
    verbo_limpio = re.sub(r'[^\w]', '', verbo.lower())
    
    if any(verbo_limpio.endswith(t) for t in ['a', 'as', 'amos', 'an', 'e', 'es', 'emos', 'en']):
        return 'Presente'
    elif any(verbo_limpio.endswith(t) for t in ['ara', 'aras', 'áramos', 'aran', 'iera', 'ieras', 'iéramos', 'ieran', 'era', 'eras', 'éramos', 'eran', 'ese', 'eses', 'ésemos', 'esen']):
        return 'Pretérito imperfecto'
    elif any(verbo_limpio.endswith(t) for t in ['are', 'ares', 'áremos', 'aren', 'iere', 'ieres', 'iéremos', 'ieren']):
        return 'Futuro simple'
    else:
        return 'Indeterminado'

def determinar_persona(verbo):
    """Determina la persona y número del verbo"""
    verbo_limpio = re.sub(r'[^\w]', '', verbo.lower())
    
    if verbo_limpio.endswith(('o', 'a', 'e')):  # 1ra singular
        return '1ra persona singular'
    elif verbo_limpio.endswith(('as', 'es')):  # 2da singular
        return '2da persona singular'
    elif verbo_limpio.endswith(('a', 'e')):  # 3ra singular
        return '3ra persona singular'
    elif verbo_limpio.endswith(('amos', 'emos', 'imos')):  # 1ra plural
        return '1ra persona plural'
    elif verbo_limpio.endswith(('áis', 'éis', 'ís')):  # 2da plural
        return '2da persona plural'
    elif verbo_limpio.endswith(('an', 'en')):  # 3ra plural
        return '3ra persona plural'
    else:
        return 'Indeterminada'

def obtener_lema_verbal(verbo):
    """Intenta obtener el lema (forma infinitiva) de un verbo"""
    verbo_limpio = re.sub(r'[^\w]', '', verbo.lower())
    
    # Para verbos irregulares, usar un diccionario
    verbos_irregulares = {
        'sea': 'ser', 'seas': 'ser', 'seamos': 'ser', 'sean': 'ser',
        'vaya': 'ir', 'vayas': 'ir', 'vayamos': 'ir', 'vayan': 'ir',
        'haya': 'haber', 'hayas': 'haber', 'hayamos': 'haber', 'hayan': 'haber',
        'esté': 'estar', 'estés': 'estar', 'estemos': 'estar', 'estén': 'estar',
        'dé': 'dar', 'des': 'dar', 'demos': 'dar', 'den': 'dar',
        'sepa': 'saber', 'sepas': 'saber', 'sepamos': 'saber', 'sepan': 'saber',
        'haga': 'hacer', 'hagas': 'hacer', 'hagamos': 'hacer', 'hagan': 'hacer',
        'pueda': 'poder', 'puedas': 'poder', 'podamos': 'poder', 'puedan': 'poder'
    }
    
    if verbo_limpio in verbos_irregulares:
        return verbos_irregulares[verbo_limpio]
    
    # Intentar stemmization si está disponible
    if st.session_state.stemmer_disponible:
        try:
            raiz = stemmer.stem(verbo_limpio)
            # Intentar determinar la terminación del infinitivo
            if verbo_limpio.endswith(('a', 'as', 'amos', 'an', 'ara', 'aras', 'áramos', 'aran', 'are', 'ares', 'áremos', 'aren')):
                return raiz + "ar"
            elif verbo_limpio.endswith(('e', 'es', 'emos', 'en', 'era', 'eras', 'éramos', 'eran', 'ere', 'eres', 'éremos', 'eren')):
                return raiz + "er"
            elif verbo_limpio.endswith(('e', 'es', 'imos', 'en', 'iera', 'ieras', 'iéramos', 'ieran', 'iere', 'ieres', 'iéremos', 'ieren')):
                return raiz + "ir"
            else:
                return raiz
        except:
            return verbo_limpio  # Si no se puede determinar, devolver la forma original
    else:
        return verbo_limpio  # Devolver la forma original si no hay stemmer

def crear_excel(resultados):
    """Crea un archivo Excel con los resultados"""
    if not resultados:
        return None
    
    df = pd.DataFrame(resultados)
    
    # Crear el archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Subjuntivos', index=False)
        
        # Obtener el libro y la hoja de trabajo para aplicar formato
        workbook = writer.book
        worksheet = writer.sheets['Subjuntivos']
        
        # Formato para los encabezados
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#366092',
            'font_color': 'white',
            'border': 1
        })
        
        # Aplicar formato a los encabezados
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Ajustar el ancho de las columnas
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    
    output.seek(0)
    return output

def crear_csv(resultados):
    """Crea un archivo CSV con los resultados"""
    if not resultados:
        return None
    
    df = pd.DataFrame(resultados)
    
    # Crear el archivo CSV en memoria
    output = BytesIO()
    
    # Escribir el CSV con codificación UTF-8 para caracteres especiales
    output.write(df.to_csv(index=False, encoding='utf-8').encode('utf-8'))
    
    output.seek(0)
    return output

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **Características:**
    - Identificación de verbos en subjuntivo
    - Análisis de tiempo y persona verbal
    - Extracción de cláusulas completas
    - Generación de informes en Excel y CSV
    
    **Ejemplos de subjuntivo:**
    - Es importante que **estudies**
    - Ojalá **llueva** mañana
    - Quiero que **vengas** pronto
    """)
    
    if not st.session_state.nltk_disponible:
        st.warning("""
        ⚠️ Algunos recursos de NLTK no están disponibles. 
        La aplicación está usando métodos alternativos para el análisis.
        """)

# Área de texto para entrada
col1, col2 = st.columns([2, 1])

with col1:
    texto = st.text_area(
        "Introduce el texto a analizar:",
        height=300,
        placeholder="Ejemplo: Es necesario que estudies más para el examen. Ojalá que tengas suerte en tu viaje..."
    )

with col2:
    st.markdown("### 📊 Estadísticas")
    if texto:
        tokens = tokenizar_texto(texto)
        total_palabras = len(tokens)
        total_oraciones = len(re.split(r'[.!?]+', texto)) if texto else 0
        
        st.metric("Palabras", total_palabras)
        st.metric("Oraciones", total_oraciones)
        
        if texto:
            verbos = [palabra for palabra in tokens if es_verbo_subjuntivo(palabra)]
            st.metric("Verbos subjuntivo", len(verbos))
    else:
        st.info("Introduce texto para ver estadísticas")

# Botón para analizar
if st.button("🔍 Analizar Subjuntivo", type="primary"):
    if not texto.strip():
        st.warning("Por favor, introduce un texto para analizar.")
    else:
        with st.spinner("Analizando texto..."):
            resultados = analizar_con_nltk(texto)
        
        if resultados:
            st.success(f"✅ Se encontraron {len(resultados)} verbos en subjuntivo")
            
            # Mostrar resultados en tabla
            st.subheader("📋 Resultados del Análisis")
            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)
            
            # Crear columnas para los botones de descarga
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Generar y descargar Excel
                excel_file = crear_excel(resultados)
                
                st.download_button(
                    label="📥 Descargar Informe Excel",
                    data=excel_file,
                    file_name="analisis_subjuntivo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_download2:
                # Generar y descargar CSV
                csv_file = crear_csv(resultados)
                
                st.download_button(
                    label="📄 Descargar Informe CSV",
                    data=csv_file,
                    file_name="analisis_subjuntivo.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Mostrar estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total subjuntivos", len(resultados))
            with col2:
                tiempos = df['Tiempo'].value_counts()
                st.metric("Tiempo más común", tiempos.index[0] if len(tiempos) > 0 else "N/A")
            with col3:
                st.metric("Verbos únicos", df['Lema'].nunique())
            
        else:
            st.info("ℹ️ No se encontraron verbos en modo subjuntivo en el texto.")

# Ejemplos predefinidos
st.subheader("💡 Ejemplos para probar")
ejemplos = {
    "Ejemplo 1": "Es importante que estudies para el examen. Ojalá que tengas buena suerte.",
    "Ejemplo 2": "Quiero que vengas a la fiesta. Dudo que ella pueda asistir.",
    "Ejemplo 3": "Sería bueno que lloviera pronto. Temo que se sequen las plantas."
}

cols = st.columns(3)
for i, (nombre, ejemplo) in enumerate(ejemplos.items()):
    with cols[i]:
        if st.button(f"📌 {nombre}"):
            texto = ejemplo
            st.rerun()

# Información adicional
with st.expander("📚 Acerca del modo subjuntivo"):
    st.markdown("""
    El modo subjuntivo en español se utiliza para expresar:
    
    - **Deseos**: Ojalá que tengas suerte
    - **Dudas**: No creo que venga
    - **Emociones**: Me alegra que estés aquí
    - **Impersonalidad**: Es necesario que estudies
    - **Consejos**: Te sugiero que leas más
    - **Hipótesis**: Si tuviera dinero, viajaría
    
    Esta aplicación detecta las formas verbales más comunes del subjuntivo,
    pero puede haber casos complejos que requieran análisis manual.
    """)

# Pie de página
st.markdown("---")
st.caption("Analizador de Modo Subjuntivo v2.0 | Desarrollado con Streamlit")
