import streamlit as st
import pandas as pd
import re
from io import BytesIO
import subprocess
import sys

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Subjuntivo Español",
    page_icon="📝",
    layout="wide"
)

# Título y descripción
st.title("🔍 Analizador de Modo Subjuntivo en Español")
st.markdown("""
Esta aplicación identifica y analiza todas las formas verbales en modo subjuntivo 
en textos en español usando métodos optimizados para el español.
""")

# Intentar instalar e importar pattern.es
try:
    from pattern.es import conjugate, lemma, lexeme, tenses, INFINITIVE
    st.session_state.pattern_available = True
except ImportError:
    st.session_state.pattern_available = False
    st.warning("""
    ⚠️ La biblioteca Pattern no está instalada. 
    La aplicación usará un método alternativo para análisis morfológico.
    """)
    
    # Botón para instalar pattern
    if st.button("🔄 Instalar Pattern.es automáticamente"):
        with st.spinner("Instalando pattern-es..."):
            try:
                # Instalar pattern-es
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pattern-es"])
                st.success("✅ Pattern.es instalado correctamente. Por favor, recarga la página.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error instalando Pattern.es: {str(e)}")

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
    'oyas', 'oiga', 'oigamos', 'oigan',  # oír
    'caiga', 'caigas', 'caigamos', 'caigan',  # caer
    'traiga', 'traigas', 'traigamos', 'traigan',  # traer
    'valga', 'valgas', 'valgamos', 'valgan',  # valer
    'salga', 'salgas', 'salgamos', 'salgan',  # salir
    'duerma', 'duermas', 'durmamos', 'duerman',  # dormir
    'muera', 'mueras', 'muramos', 'mueran',  # morir
    'sienta', 'sientas', 'sintamos', 'sientan',  # sentir
    'pida', 'pidas', 'pidamos', 'pidan',  # pedir
    'cuente', 'cuentes', 'contemos', 'cuenten',  # contar
    'vuelva', 'vuelvas', 'volvamos', 'vuelvan',  # volver
    'encuentre', 'encuentres', 'encontremos', 'encuentren'  # encontrar
]

# Conectores que suelen introducir subjuntivo
conectores_subjuntivo = [
    'que', 'cuando', 'si', 'aunque', 'para que', 'a fin de que', 
    'como si', 'a menos que', 'con tal de que', 'en caso de que',
    'sin que', 'antes de que', 'ojalá', 'espero que', 'dudo que',
    'no creo que', 'es posible que', 'es probable que', 'quizás',
    'tal vez', 'a no ser que', 'salvo que', 'excepto que',
    'mientras', 'después de que', 'hasta que', 'en cuanto',
    'siempre que', 'por más que', 'a pesar de que'
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

def es_verbo_subjuntivo(palabra):
    """Determina si una palabra es un verbo en subjuntivo usando pattern.es si está disponible"""
    palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())
    
    if not palabra_limpia:
        return False
    
    # 1. Verificar verbos irregulares
    if palabra_limpia in verbos_irregulares_subjuntivo:
        return True
    
    # 2. Verificar por terminaciones típicas del subjuntivo
    for terminacion in subjuntivo_terminaciones:
        if palabra_limpia.endswith(terminacion):
            return True
    
    # 3. Usar pattern.es si está disponible para análisis más preciso
    if st.session_state.pattern_available:
        try:
            # Obtener todos los tiempos verbales de esta forma
            tiempos_verbales = tenses(palabra_limpia)
            for tiempo in tiempos_verbales:
                # El modo subjuntivo se representa como 'subjunctive' en pattern
                if 'subjunctive' in str(tiempo).lower():
                    return True
        except:
            # Si pattern falla, continuar con otros métodos
            pass
    
    return False

def obtener_lema_verbal(palabra):
    """Obtiene el lema (infinitivo) de un verbo"""
    palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())
    
    # Diccionario de verbos irregulares
    verbos_irregulares = {
        'sea': 'ser', 'seas': 'ser', 'seamos': 'ser', 'sean': 'ser',
        'vaya': 'ir', 'vayas': 'ir', 'vayamos': 'ir', 'vayan': 'ir',
        'haya': 'haber', 'hayas': 'haber', 'hayamos': 'haber', 'hayan': 'haber',
        'esté': 'estar', 'estés': 'estar', 'estemos': 'estar', 'estén': 'estar',
        'dé': 'dar', 'des': 'dar', 'demos': 'dar', 'den': 'dar',
        'sepa': 'saber', 'sepas': 'saber', 'sepamos': 'saber', 'sepan': 'saber',
        'haga': 'hacer', 'hagas': 'hacer', 'hagamos': 'hacer', 'hagan': 'hacer',
        'pueda': 'poder', 'puedas': 'poder', 'podamos': 'poder', 'puedan': 'poder',
        'quiera': 'querer', 'quieras': 'querer', 'queramos': 'querer', 'quieran': 'querer',
        'tenga': 'tener', 'tengas': 'tener', 'tengamos': 'tener', 'tengan': 'tener',
        'venga': 'venir', 'vengas': 'venir', 'vengamos': 'venir', 'vengan': 'venir'
    }
    
    if palabra_limpia in verbos_irregulares:
        return verbos_irregulares[palabra_limpia]
    
    # Usar pattern.es si está disponible
    if st.session_state.pattern_available:
        try:
            lema_verbo = lemma(palabra_limpia)
            if lema_verbo and lema_verbo != palabra_limpia:
                return lema_verbo
        except:
            pass
    
    # Método de respaldo: inferir el infinitivo desde la terminación
    if palabra_limpia.endswith(('a', 'as', 'amos', 'an', 'ara', 'aras', 'áramos', 'aran', 'are', 'ares', 'áremos', 'aren')):
        return palabra_limpia[:-2] + 'ar' if len(palabra_limpia) > 2 else palabra_limpia + 'ar'
    elif palabra_limpia.endswith(('e', 'es', 'emos', 'en', 'era', 'eras', 'éramos', 'eran', 'ere', 'eres', 'éremos', 'eren')):
        return palabra_limpia[:-2] + 'er' if len(palabra_limpia) > 2 else palabra_limpia + 'er'
    elif palabra_limpia.endswith(('e', 'es', 'imos', 'en', 'iera', 'ieras', 'iéramos', 'ieran', 'iere', 'ieres', 'iéremos', 'ieren')):
        return palabra_limpia[:-2] + 'ir' if len(palabra_limpia) > 2 else palabra_limpia + 'ir'
    
    return palabra_limpia

def determinar_tiempo_verbal(verbo):
    """Determina el tiempo verbal aproximado"""
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

def encontrar_clausula_subjuntivo(texto, posicion_verbo):
    """Encuentra la cláusula que contiene el verbo en subjuntivo"""
    # Buscar hacia atrás para encontrar el inicio de la cláusula
    inicio = max(0, posicion_verbo - 100)  # Buscar hasta 100 caracteres atrás
    
    for conector in conectores_subjuntivo:
        idx = texto.rfind(conector, inicio, posicion_verbo)
        if idx != -1:
            inicio = idx
            break
    
    # Buscar hacia adelante para encontrar el final de la cláusula
    fin = min(len(texto), posicion_verbo + 100)  # Buscar hasta 100 caracteres adelante
    
    for puntuacion in ['.', '!', '?', ';']:
        idx = texto.find(puntuacion, posicion_verbo)
        if idx != -1 and idx < fin:
            fin = idx + 1
            break
    
    return texto[inicio:fin].strip()

def analizar_texto(texto):
    """Analiza el texto para identificar verbos en subjuntivo"""
    # Usar una expresión regular para encontrar palabras (incluyendo acentos)
    palabras = re.findall(r'\b[a-záéíóúñ]+\b', texto.lower())
    posiciones = []
    
    # Encontrar todas las posiciones de las palabras
    for match in re.finditer(r'\b[a-záéíóúñ]+\b', texto.lower()):
        posiciones.append((match.group(), match.start()))
    
    resultados = []
    
    for palabra, posicion in posiciones:
        if es_verbo_subjuntivo(palabra):
            # Encontrar la cláusula
            clausula = encontrar_clausula_subjuntivo(texto, posicion)
            
            # Determinar tiempo verbal aproximado
            tiempo = determinar_tiempo_verbal(palabra)
            
            # Determinar persona y número
            persona = determinar_persona(palabra)
            
            # Obtener lema (forma infinitiva)
            lema_verbo = obtener_lema_verbal(palabra)
            
            resultados.append({
                'Verbo': texto[posicion:posicion+len(palabra)],
                'Lema': lema_verbo,
                'Tiempo': tiempo,
                'Persona': persona,
                'Cláusula': clausula,
                'Posición': f"Carácter {posicion}"
            })
    
    return resultados

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
    - Identificación precisa de verbos en subjuntivo
    - Análisis de tiempo y persona verbal
    - Extracción de cláusulas completas
    - Generación de informes en Excel y CSV
    
    **Tecnología:**
    - Optimizado específicamente para español
    - Patrones morfológicos avanzados
    - Diccionario extenso de verbos irregulares
    """)
    
    if st.session_state.pattern_available:
        st.success("✅ Pattern.es está disponible para análisis avanzado")
    else:
        st.warning("⚠️ Usando método alternativo (Pattern.es no disponible)")

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
        # Contar palabras (considerando acentos españoles)
        palabras = re.findall(r'\b[a-záéíóúñ]+\b', texto.lower())
        total_palabras = len(palabras)
        total_oraciones = len(re.split(r'[.!?]+', texto))
        
        st.metric("Palabras", total_palabras)
        st.metric("Oraciones", total_oraciones)
        
        # Contar verbos en subjuntivo aproximados
        verbos_subjuntivo = [p for p in palabras if es_verbo_subjuntivo(p)]
        st.metric("Verbos subjuntivo", len(verbos_subjuntivo))
    else:
        st.info("Introduce texto para ver estadísticas")

# Botón para analizar
if st.button("🔍 Analizar Subjuntivo", type="primary"):
    if not texto.strip():
        st.warning("Por favor, introduce un texto para analizar.")
    else:
        with st.spinner("Analizando texto..."):
            resultados = analizar_texto(texto)
        
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
    
    **Tiempos verbales del subjuntivo:**
    - Presente: que hable, que comas, que vivamos
    - Pretérito imperfecto: que hablara/hablase, que comieras/comieses
    - Futuro simple: que hablare, que comieres (poco usado)
    """)

# Pie de página
st.markdown("---")
st.caption("Analizador de Modo Subjuntivo v3.0 | Optimizado para español")
