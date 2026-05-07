import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Configuración estética de la página
st.set_page_config(page_title="Glow Up Search", page_icon="✨")

st.title("✨ IT-GIRL Style & Beauty Finder")

st.write("""
¿No sabes qué ponerte o qué rutina seguir? Deja que la **IA** encuentre tu match ideal. 🎀

1️⃣ Escribe tus opciones de outfits o tips de belleza.  
2️⃣ Cuéntale al sistema cómo te sientes hoy o qué buscas.  
3️⃣ ¡Descubre tu **Perfect Match** al instante! 💖
""")

# Document input (ahora son tips o outfits)
text_input = st.text_area(
    "Tu clóset digital o diario de belleza (uno por línea):",
    """Vestido satinado rosa con tacones altos para una cena romántica.
Outfit deportivo aesthetic con leggings flare y top blanco para el gym.
Rutina de skincare coreana con doble limpieza y mucha hidratación para una piel de cristal.
Jeans baggy con oversize hoodie y sneakers retro para un look urbano y comfy."""
)

question = st.text_input(
    "¿Cuál es el mood de hoy, reina?",
    "Busco algo cómodo pero con estilo para salir a caminar."
)

if st.button("✨ Encontrar mi Vibe ✨"):

    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) == 0:
        st.warning("¡Oops! Olvidaste escribir tus opciones de estilo. 🌸")

    else:
        # TF-IDF (El cerebro detrás del estilo)
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Opción {i+1}" for i in range(len(documents))]
        )

        with st.expander("Ver análisis técnico del estilo (TF-IDF Matrix) 🔬"):
            st.dataframe(df_tfidf.round(3))

        # Question vector
        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.divider()
        st.subheader("💖 ¡Tu Match Ideal!")

        st.write(f"**Tu mood:** {question}")
        st.write(f"**Recomendación seleccionada (Opción {best_idx+1}):**")
        
        # Un toque de estilo al resultado
        st.success(f"✨ {best_doc}")

        if best_score > 0:
            st.write(f"Nivel de compatibilidad: **{best_score*100:.1f}% Match**")
        else:
            st.info("No encontramos un match exacto, ¡pero podrías intentar algo nuevo! 💅")

        # Similarity ranking
        sim_df = pd.DataFrame({
            "Opción": [f"Nº {i+1}" for i in range(len(documents))],
            "Sugerencia": documents,
            "Compatibilidad": similarities
        })

        st.subheader("Otras opciones para ti 🛍️")
        st.dataframe(sim_df.sort_values("Compatibilidad", ascending=False))

        # Mostrar palabras clave importantes
        st.subheader("Etiquetas clave de tu búsqueda 🏷️")

        feature_names = vectorizer.get_feature_names_out()

        for i, row in enumerate(X.toarray()):
            top_indices = row.argsort()[-5:][::-1]
            top_words = [feature_names[j] for j in top_indices if row[j] > 0]
            
            if top_words:
                st.write(f"**Opción {i+1} vibes:** {', '.join(top_words)} ✨")

st.markdown("---")
st.caption("Hecho con ✨ y Python para chicas techies.")




