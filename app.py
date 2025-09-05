import streamlit as st
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import plotly.express as px

# Descargar recursos NLTK (solo una vez)
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    return True

download_nltk()

# Diccionario base de formas subjuntivas comunes
subjunctive_forms = {
    'presente': {
        'ar': [r'[aei]ría\w*', r'[aei]ré\w*', r'[aei]rí\w*'],
        'er': [r'[aei]ríe\w*', r'[aei]ré\w*', r'[aei]rí\w*'],
        'ir': [r'[aei]rí\w*', r'[aei]ré\w*', r'[aei]rí\w*']
    },
    'imperfecto': {
        '1ra': [r'[aei]r[aá]ra\w*', r'[aei]rí\w*'],
        '2da': [r'[aei]r[eé]se\w*', r'[aei]rí\w*']
    },
    'pluscuamperfecto': [r'hubier[ea]\w*', r'hici[ea]\w*']
}

subjunctive_triggers = [
    'ojalá', 'espero que', 'quizás', 'quizá', 'tal vez', 'aunque', 'a pesar de que',
    'para que', 'sin que', 'antes de que', 'mientras', 'como si', 'como',
    'si', 'en caso de que', 'cuando', 'donde', 'cual', 'cuyo', 'cuyos', 'cuya', 'cuyas',
    'que', 'no creo que', 'me parece que', 'creo que', 'pienso que', 'dudo que'
]

def detect_subjunctive(text):
    text = text.lower()
    sentences = sent_tokenize(text, language='spanish')
    
    results = {
        'subjunctive_verbs': [],
        'contexts': defaultdict(list),
        'summary': {}
    }
    
    for sent in sentences:
        tokens = word_tokenize(sent, language='spanish')
        pos_tags = pos_tag(tokens)
        
        trigger_found = any(trigger in sent for trigger in subjunctive_triggers)
        
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('V'):
                analysis = analyze_verb(word, sent, trigger_found)
                if analysis['is_subjunctive']:
                    verb_data = {
                        'verbo': word,
                        'forma': analysis['form'],
                        'tiempo': analysis['tense'].capitalize(),
                        'persona': analysis['person'],
                        'numero': analysis['number'],
                        'oracion': sent,
                        'posicion': i
                    }
                    
                    results['subjunctive_verbs'].append(verb_data)
                    results['contexts'][verb_data['tiempo']].append(verb_data)
    
    results['summary'] = generate_summary(results)
    return results

def analyze_verb(verb, sentence, trigger_found):
    analysis = {
        'is_subjunctive': False,
        'form': None,
        'tense': None,
        'person': None,
        'number': None
    }
    
    # Verificación directa
    for tense, forms in subjunctive_forms.items():
        if tense == 'presente':
            for ending, patterns in forms.items():
                for pattern in patterns:
                    if re.search(pattern, verb):
                        analysis['is_subjunctive'] = True
                        analysis['tense'] = 'presente'
                        analysis['form'] = 'subjuntivo'
                        analysis['person'] = get_person(verb, tense)
                        analysis['number'] = get_number(verb, tense)
                        return analysis
        elif tense == 'imperfecto':
            for form_name, patterns in forms.items():
                for pattern in patterns:
                    if re.search(pattern, verb):
                        analysis['is_subjunctive'] = True
                        analysis['tense'] = 'imperfecto'
                        analysis['form'] = f'subjuntivo_{form_name}'
                        analysis['person'] = get_person(verb, tense)
                        analysis['number'] = get_number(verb, tense)
                        return analysis
        elif tense == 'pluscuamperfecto':
            for pattern in forms:
                if re.search(pattern, verb):
                    analysis['is_subjunctive'] = True
                    analysis['tense'] = 'pluscuamperfecto'
                    analysis['form'] = 'subjuntivo'
                    analysis['person'] = '3a'
                    analysis['number'] = 'singular'
                    return analysis
    
    # Heurísticas contextuales
    if trigger_found:
        subjunctive_endings = [
            r'r[aeí]a\w*', r'r[aeí]e\w*', r'rí\w*', r'ría\w*'
        ]
        
        for ending in subjunctive_endings:
            if re.search(ending, verb):
                if not re.match(r'.*ría$', verb) or 'querría' not in sentence:
                    analysis['is_subjunctive'] = True
                    analysis['tense'] = 'presente' if 'ría' in verb else 'imperfecto'
                    analysis['form'] = 'subjuntivo'
                    analysis['person'] = get_person(verb, 'contextual')
                    analysis['number'] = get_number(verb, 'contextual')
                    return analysis
    
    return analysis

def get_person(verb, tense):
    if tense == 'presente':
        if verb.endswith(('e', 'es', 'é', 'és')):
            return '3a' if verb.endswith('e') else '2a'
        elif verb.endswith(('a', 'as', 'á', 'ás')):
            return '3a' if verb.endswith('a') else '2a'
        elif verb.endswith(('emos', 'éis', 'en')):
            return '1a' if 'emos' in verb else '2a' if 'éis' in verb else '3a'
    elif tense == 'imperfecto':
        if verb.endswith(('ra', 'ras', 'ramos', 'rais', 'ran')):
            return '3a' if verb.endswith(('ra', 'ran')) else '2a' if verb.endswith(('ras', 'rais')) else '1a'
        elif verb.endswith(('se', 'ses', 'semos', 'seis', 'sen')):
            return '3a' if verb.endswith(('se', 'sen')) else '2a' if verb.endswith(('ses', 'seis')) else '1a'
    elif tense == 'pluscuamperfecto':
        return '3a'
    return 'indeterminada'

def get_number(verb, tense):
    if tense == 'presente':
        if verb.endswith(('e', 'a', 'é', 'á')):
            return 'singular'
        elif verb.endswith(('en', 'es', 'as', 'éis', 'emos')):
            return 'plural'
    elif tense == 'imperfecto':
        if verb.endswith(('ra', 'se', 'ras', 'ses')):
            return 'singular' if verb.endswith(('ra', 'se')) else 'plural'
        elif verb.endswith(('ramos', 'rais', 'ran', 'semos', 'seis', 'sen')):
            return 'plural'
    return 'singular'

def generate_summary(results):
    summary = {
        'total_verbs': len(results['subjunctive_verbs']),
        'tenses': {},
        'most_common': []
    }
    
    tense_count = defaultdict(int)
    for verb in results['subjunctive_verbs']:
        tense = verb['tiempo']
        tense_count[tense] += 1
    
    for tense, count in tense_count.items():
        summary['tenses'][tense] = count
    
    verb_freq = defaultdict(int)
    for verb in results['subjunctive_verbs']:
        verb_freq[verb['verbo']] += 1
    
    summary['most_common'] = sorted(verb_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return summary

# Interfaz Streamlit
st.set_page_config(
    page_title="Detector de Subjuntivo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Detector de Subjuntivo en Español")
st.markdown("**Analiza automáticamente verbos en modo subjuntivo en textos en español**")

with st.sidebar:
    st.header("Acerca de")
    st.markdown("""
    Esta aplicación utiliza procesamiento de lenguaje natural para:
    
    - ✅ Detectar verbos en subjuntivo (presente, imperfecto, pluscuamperfecto)
    - ✅ Identificar contextos que requieren subjuntivo
    - ✅ Mostrar análisis morfológico (persona, número, tiempo)
    - ✅ Proporcionar estadísticas de uso
    
    *Ideal para estudiantes de español y corrección de textos.*
    """)
    
    st.markdown("---")
    st.markdown("**Ejemplos de uso**")
    example_text = "Ojalá llueva mañana. Espero que tengas un buen día. Aunque esté cansado, iré al trabajo. No creo que haya suficiente tiempo para terminar."
    if st.button("Cargar ejemplo"):
        st.session_state.text_input = example_text

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "Introduce tu texto en español:",
        value=st.session_state.get('text_input', ''),
        height=300,
        placeholder="Pega aquí tu texto para analizar..."
    )
    
    analyze_button = st.button("🔍 Analizar Texto", type="primary")

with col2:
    st.markdown("### 📊 Resumen Rápido")
    if 'results' in st.session_state:
        results = st.session_state.results
        summary = results['summary']
        
        total = summary['total_verbs']
        st.metric("Verbos en subjuntivo", total, 
                 delta="✅ Correcto" if total > 0 else "❌ Ninguno", 
                 delta_color="normal")
        
        if total > 0:
            tenses = summary['tenses']
            for tense, count in tenses.items():
                st.metric(tense, count)
    else:
        st.info("El análisis aparecerá aquí")

# Procesamiento
if analyze_button and text:
    with st.spinner('Analizando texto...'):
        try:
            results = detect_subjunctive(text)
            st.session_state.results = results
            st.session_state.text_input = text
            
            # Mostrar resultados
            summary = results['summary']
            total = summary['total_verbs']
            
            if total == 0:
                st.success("✅ No se encontraron verbos en subjuntivo en el texto.")
            else:
                st.success(f"✅ Se encontraron **{total}** verbos en subjuntivo")
                
                # Gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución por tiempo verbal
                    tense_data = pd.DataFrame(list(summary['tenses'].items()), 
                                            columns=['Tiempo', 'Cantidad'])
                    fig_tense = px.bar(tense_data, x='Tiempo', y='Cantidad', 
                                     title='Distribución por Tiempo Verbal',
                                     color='Cantidad', color_continuous_scale='Blues')
                    st.plotly_chart(fig_tense, use_container_width=True)
                
                with col2:
                    # Verbos más comunes
                    verb_data = pd.DataFrame(summary['most_common'], 
                                           columns=['Verbo', 'Frecuencia'])
                    fig_verb = px.bar(verb_data, x='Verbo', y='Frecuencia', 
                                    title='Verbos más frecuentes',
                                    color='Frecuencia', color_continuous_scale='Reds')
                    st.plotly_chart(fig_verb, use_container_width=True)
                
                # Tabla de resultados
                st.markdown("### 📋 Detalle de verbos encontrados")
                df = pd.DataFrame(results['subjunctive_verbs'])
                df = df[['verbo', 'tiempo', 'persona', 'numero', 'oracion']]
                df.columns = ['Verbo', 'Tiempo', 'Persona', 'Número', 'Oración']
                
                # Formatear la tabla
                def highlight_verbs(val):
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                
                styled_df = df.style.applymap(highlight_verbs, subset=['Verbo'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Contexto detallado
                st.markdown("### 🔍 Contexto detallado")
                for i, verb in enumerate(results['subjunctive_verbs'], 1):
                    with st.expander(f"**{i}. {verb['verbo']}** ({verb['tiempo']})"):
                        st.markdown(f"""
                        - **Forma:** {verb['forma']}
                        - **Persona:** {verb['persona']}
                        - **Número:** {verb['numero']}
                        - **Oración:** *"{verb['oracion']}"*
                        """)
                        
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.info("Por favor, intenta con un texto diferente o más corto.")

# Información adicional
st.markdown("---")
st.markdown("### 📝 Notas sobre el análisis")
st.info("""
- La aplicación detecta formas regulares e irregulares de subjuntivo
- Considera contextos que suelen introducir subjuntivo (ojalá, espero que, etc.)
- Puede haber falsos positivos en formas verbales ambiguas
- Para mejores resultados, utiliza textos completos con contexto
""")

st.markdown("### 📚 Recursos útiles")
st.markdown("""
- [Gramática del subjuntivo en español](https://www.rae.es)
- [Ejercicios de subjuntivo](https://www.studyspanish.com)
- [Conjugador de verbos](https://www.spanishdict.com)
""")
