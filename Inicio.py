import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. Configuración de página (Esto siempre debe ir primero)
st.set_page_config(page_title="Glow Up Search", page_icon="✨", layout="centered")

# 2. Estilo CSS Inyectado para que se vea ROSITA de verdad
st.markdown("""
    <style>
    .stApp { background-color: #FFF5F7; }
    h1 { color: #FF85A1 !important; font-family: 'Georgia', serif; }
    .stButton>button { 
        background-color: #FFC0CB !important; 
        color: white !important; 
        border-radius: 20px !important;
        border: 2px solid #FF85A1 !important;
        width: 100%;
    }
    .stTextArea textarea { border: 2px solid #FFD1DC !important; border-radius: 15px; }
    .stTextInput input { border: 2px solid #FFD1DC !important; border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ---
st.title("✨ IT-GIRL Style & Beauty Finder")

st.markdown("""
<div style="text-align: center; color: #7A4A58;">
    ¿No sabes qué ponerte o qué rutina seguir? Deja que la <b>IA</b> encuentre tu match ideal. 🎀
</div>
""", unsafe_allow_html=True)

st.write("") # Espacio

# --- ENTRADAS (Organizadas en columnas) ---
with st.container():
    st.subheader("🛍️ Tu Diario de Estilo")
    text_input = st.text_area(
        "Ingresa tus opciones (outfits, skincare, makeup) una por línea:",
        """Vestido satinado rosa con tacones altos para una cena romántica.
Outfit deportivo aesthetic con leggings flare y top blanco para el gym.
Rutina de skincare coreana con doble limpieza y mucha hidratación para una piel de cristal.
Jeans baggy con oversize hoodie y sneakers retro para un look urbano y comfy.""",
        height=150
    )

    question = st.text_input(
        "🌸 ¿Cuál es el mood de hoy, reina?",
        placeholder="Ej: Busco algo relax para un café..."
    )

# --- LÓGICA ---
if st.button("✨ Encontrar mi Vibe ✨"):
    
    # Limpiamos los documentos
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if not documents:
        st.warning("¡Oops! Olvidaste escribir tus opciones de estilo. 🌸")
    elif not question:
        st.error("Dime qué buscas hoy para poder ayudarte, nena. 💕")
    else:
        # TF-IDF Magic
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)
        question_vec = vectorizer.transform([question])

        # Similitud
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        # --- RESULTADOS ---
        st.divider()
        
        if best_score > 0:
            st.subheader("💖 ¡Tu Match Ideal Encontrado!")
            
            # Caja de resultado destacada
            st.success(f"**Recomendación:** {documents[best_idx]}")
            st.write(f"Nivel de match: **{best_score*100:.1f}%**")
            
            # Columnas para detalles adicionales
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏷️ Vibes Detectadas")
                feature_names = vectorizer.get_feature_names_out()
                top_indices = X[best_idx].toarray()[0].argsort()[-3:][::-1]
                for idx in top_indices:
                    if X[best_idx, idx] > 0:
                        st.write(f"✨ {feature_names[idx]}")

            with col2:
                st.markdown("### 🛍️ Ranking")
                sim_df = pd.DataFrame({
                    "Opción": [f"Nº {i+1}" for i in range(len(documents))],
                    "Match": similarities
                }).sort_values("Match", ascending=False)
                st.dataframe(sim_df)
        else:
            st.info("No encontré un match exacto... ¡Pero recuerda que tú creas tu propio estilo! 💅")

        # Expander técnico al final (menos relevante para el look)
        with st.expander("Ver análisis técnico (TF-IDF Matrix) 🔬"):
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Opción {i+1}" for i in range(len(documents))]
            )
            st.dataframe(df_tfidf.round(3))

st.markdown("---")
st.caption("Hecho con ✨ y Python para chicas techies.")




