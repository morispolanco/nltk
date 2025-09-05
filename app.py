import streamlit as st
import pandas as pd
import re
from io import BytesIO
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import cess_esp
from nltk.stem import SnowballStemmer

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/cess_esp')
except LookupError:
    nltk.download('cess_esp')

try:
    nltk.data.find('taggers/cess_esp')
except LookupError:
    nltk.download('cess_esp_udep')

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

# Inicializar el stemmer en español
stemmer = SnowballStemmer("spanish")

# Cargar el tagset de cess_esp para español
try:
    # Obtener el tagged corpus
    tagged_sents = cess_esp.tagged_sents()
    st.session_state.corpus_cargado = True
except:
    st.session_state.corpus_cargado = False
    st.warning("El corpus CESS_ESP no está disponible. Algunas funciones avanzadas estarán limitadas.")

# Diccionario de etiquetas POS (Part-of-Speech) para español
POS_TAGS = {
    'ao': 'Adjetivo ordinal',
    'aq': 'Adjetivo calificativo',
    'cc': 'Conjunción coordinada',
    'cs': 'Conjunción subordinada',
    'da': 'Determinante artículo',
    'dd': 'Determinante demostrativo',
    'de': 'Determinante exclamativo',
    'di': 'Determinante indefinido',
    'dn': 'Determinante numeral',
    'do': 'Determinante posesivo',
    'dt': 'Determinante interrogativo',
    'f': 'Puntuación',
    'i': 'Interjección',
    'nc': 'Nombre común',
    'np': 'Nombre propio',
    'p': 'Preposición',
    'pd': 'Pronombre demostrativo',
    'pe': 'Pronombre exclamativo',
    'pi': 'Pronombre indefinido',
    'pn': 'Pronombre numeral',
    'pp': 'Pronombre personal',
    'pr': 'Pronombre relativo',
    'pt': 'Pronombre interrogativo',
    'px': 'Pronombre posesivo',
    'rg': 'Adverbio general',
    'rn': 'Adverbio de negación',
    'sp': 'Adposición',
    'va': 'Verbo auxiliar',
    'vm': 'Verbo principal',
    'vs': 'Verbo semiauxiliar',
    'w': 'Fecha',
    'z': 'Numeral',
    'zm': 'Numeral monetario'
}

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

def analizar_con_nltk(texto):
    """Analiza el texto usando NLTK para identificar verbos en subjuntivo"""
    # Tokenizar y etiquetar
    tokens = word_tokenize(texto, language='spanish')
    tagged = pos_tag(tokens)
    
    resultados = []
    
    for i, (palabra, tag) in enumerate(tagged):
        if es_verbo_subjuntivo(palabra, tag):
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
                'Etiqueta': tag,
                'Tiempo': tiempo,
                'Persona': persona,
                'Cláusula': clausula,
                'Posición': f"Token {i+1}"
            })
    
    return resultados

def es_verbo_subjuntivo(palabra, tag):
    """Determina si una palabra es un verbo en subjuntivo basado en etiquetas POS y formas verbales"""
    # Verificar si es verbo según la etiqueta POS
    if not tag.startswith('v'):
        return False
    
    palabra = palabra.lower()
    
    # Verificar verbos irregulares
    if palabra in verbos_irregulares_subjuntivo:
        return True
    
    # Verificar por terminaciones típicas del subjuntivo
    terminaciones_subjuntivo = [
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
    
    for terminacion in terminaciones_subjuntivo:
        if palabra.endswith(terminacion):
            return True
    
    return False

def encontrar_clausula_subjuntivo(tokens, posicion_verbo):
    """Encuentra la cláusula que contiene el verbo en subjuntivo"""
    # Buscar hacia atrás para encontrar el inicio de la cláusula
    inicio = 0
    for i in range(posicion_verbo, 0, -1):
        if tokens[i].lower() in conectores_subjuntivo:
            inicio = i
            break
    
    # Buscar hacia adelante para encontrar el final de la cláusula
    fin = len(tokens)
    for i in range(posicion_verbo, len(tokens)):
        if tokens[i] in ['.', '!', '?', ';']:
            fin = i + 1
            break
    
    # Construir la cláusula
    clausula = ' '.join(tokens[inicio:fin])
    return clausula

def determinar_tiempo_verbal(verbo):
    """Determina el tiempo verbal aproximado basado en la terminación"""
    verbo = verbo.lower()
    
    if any(verbo.endswith(t) for t in ['a', 'as', 'amos', 'an', 'e', 'es', 'emos', 'en']):
        return 'Presente'
    elif any(verbo.endswith(t) for t in ['ara', 'aras', 'áramos', 'aran', 'iera', 'ieras', 'iéramos', 'ieran', 'era', 'eras', 'éramos', 'eran', 'ese', 'eses', 'ésemos', 'esen']):
        return 'Pretérito imperfecto'
    elif any(verbo.endswith(t) for t in ['are', 'ares', 'áremos', 'aren', 'iere', 'ieres', 'iéremos', 'ieren']):
        return 'Futuro simple'
    else:
        return 'Indeterminado'

def determinar_persona(verbo):
    """Determina la persona y número del verbo"""
    verbo = verbo.lower()
    
    if verbo.endswith(('o', 'a', 'e')):  # 1ra singular
        return '1ra persona singular'
    elif verbo.endswith(('as', 'es')):  # 2da singular
        return '2da persona singular'
    elif verbo.endswith(('a', 'e')):  # 3ra singular
        return '3ra persona singular'
    elif verbo.endswith(('amos', 'emos', 'imos')):  # 1ra plural
        return '1ra persona plural'
    elif verbo.endswith(('áis', 'éis', 'ís')):  # 2da plural
        return '2da persona plural'
    elif verbo.endswith(('an', 'en')):  # 3ra plural
        return '3ra persona plural'
    else:
        return 'Indeterminada'

def obtener_lema_verbal(verbo):
    """Intenta obtener el lema (forma infinitiva) de un verbo"""
    verbo = verbo.lower()
    
    # Mapeo de terminaciones a infinitivos
    terminaciones_a_infinitivo = {
        'o': 'ar', 'as': 'ar', 'a': 'ar', 'amos': 'ar', 'an': 'ar',
        'o': 'er', 'es': 'er', 'e': 'er', 'emos': 'er', 'en': 'er',
        'o': 'ir', 'es': 'ir', 'e': 'ir', 'imos': 'ir', 'en': 'ir',
        'é': 'ar', 'aste': 'ar', 'ó': 'ar', 'amos': 'ar', 'aron': 'ar',
        'í': 'er', 'iste': 'er', 'ió': 'er', 'imos': 'er', 'ieron': 'er',
        'í': 'ir', 'iste': 'ir', 'ió': 'ir', 'imos': 'ir', 'ieron': 'ir'
    }
    
    # Para verbos irregulares, usar un diccionario
    verbos_irregulares = {
        'sea': 'ser', 'seas': 'ser', 'seamos': 'ser', 'sean': 'ser',
        'vaya': 'ir', 'vayas': 'ir', 'vayamos': 'ir', 'vayan': 'ir',
        'haya': 'haber', 'hayas': 'haber', 'hayamos': 'haber', 'hayan': 'haber',
        'esté': 'estar', 'estés': 'estar', 'estemos': 'estar', 'estén': 'estar',
        'dé': 'dar', 'des': 'dar', 'demos': 'dar', 'den': 'dar',
        'sepa': 'saber', 'sepas': 'saber', 'sepamos': 'saber', 'sepan': 'saber'
    }
    
    if verbo in verbos_irregulares:
        return verbos_irregulares[verbo]
    
    # Intentar stemmization
    try:
        return stemmer.stem(verbo) + "ar"  # Aproximación
    except:
        return verbo  # Si no se puede determinar, devolver la forma original

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
    **Características con NLTK:**
    - Tokenización y etiquetado POS en español
    - Identificación precisa de verbos
    - Análisis morfológico avanzado
    - Detección de lemas verbales
    
    **Ejemplos de subjuntivo:**
    - Es importante que **estudies**
    - Ojalá **llueva** mañana
    - Quiero que **vengas** pronto
    """)
    
    if not st.session_state.corpus_cargado:
        st.warning("""
        ⚠️ El corpus CESS_ESP no está disponible completamente. 
        Algunas funciones avanzadas de NLTK podrían estar limitadas.
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
        tokens = word_tokenize(texto, language='spanish') if texto else []
        total_palabras = len(tokens)
        total_oraciones = len(re.split(r'[.!?]+', texto)) if texto else 0
        
        st.metric("Palabras", total_palabras)
        st.metric("Oraciones", total_oraciones)
        
        if texto:
            tagged = pos_tag(tokens)
            verbos = [word for word, tag in tagged if tag.startswith('v')]
            st.metric("Verbos totales", len(verbos))
    else:
        st.info("Introduce texto para ver estadísticas")

# Botón para analizar
if st.button("🔍 Analizar Subjuntivo con NLTK", type="primary"):
    if not texto.strip():
        st.warning("Por favor, introduce un texto para analizar.")
    else:
        with st.spinner("Analizando texto con NLTK..."):
            resultados = analizar_con_nltk(texto)
        
        if resultados:
            st.success(f"✅ Se encontraron {len(resultados)} verbos en subjuntivo")
            
            # Mostrar resultados en tabla
            st.subheader("📋 Resultados del Análisis con NLTK")
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
                    file_name="analisis_subjuntivo_nltk.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_download2:
                # Generar y descargar CSV
                csv_file = crear_csv(resultados)
                
                st.download_button(
                    label="📄 Descargar Informe CSV",
                    data=csv_file,
                    file_name="analisis_subjuntivo_nltk.csv",
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
            
            # Mostrar información sobre etiquetas POS
            with st.expander("📊 Distribución de etiquetas POS"):
                distribucion = df['Etiqueta'].value_counts()
                st.bar_chart(distribucion)
                
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
with st.expander("📚 Acerca del análisis con NLTK"):
    st.markdown("""
    Esta aplicación utiliza el **Natural Language Toolkit (NLTK)** para:
    
    - **Tokenización**: Dividir el texto en palabras y oraciones
    - **Etiquetado POS**: Identificar las categorías gramaticales de cada palabra
    - **Stemming**: Reducir palabras a su raíz o lema
    
    **Ventajas de usar NLTK:**
    - Análisis lingüístico más preciso
    - Identificación de estructuras gramaticales
    - Detección de relaciones entre palabras
    
    **Limitaciones:**
    - El corpus en español de NLTK es más limitado que el de inglés
    - Algunos verbos irregulares pueden no detectarse correctamente
    """)

# Pie de página
st.markdown("---")
st.caption("Analizador de Modo Subjuntivo con NLTK v2.0 | Desarrollado con Streamlit y NLTK")
