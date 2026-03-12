import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Demo interactiva de TF-IDF con búsqueda semántica")

st.write("""
Cada línea se trata como un **documento**.  
La aplicación usa **TF-IDF + Cosine Similarity** para encontrar el documento más relevante.

✨ Nuevas funciones:
- Heatmap visual de TF-IDF
- Gráfico de similitud
- Palabras de la pregunta resaltadas en el documento
""")

# Entrada de texto
text_input = st.text_area(
    "Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Escribe una pregunta:", "Who is playing?")

stemmer = SnowballStemmer("english")


def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems


def highlight_words(text, stems):
    words = text.split()
    highlighted = []
    for w in words:
        stem = stemmer.stem(w.lower())
        if stem in stems:
            highlighted.append(f"<span style='background-color:yellow'>{w}</span>")
        else:
            highlighted.append(w)
    return " ".join(highlighted)


if st.button("Analizar documentos"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) == 0:
        st.warning("⚠️ Ingresa al menos un documento")

    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.subheader("Matriz TF-IDF")
        st.dataframe(df_tfidf.round(3))

        # 📊 Heatmap
        st.subheader("Visualización TF-IDF (Heatmap)")
        fig, ax = plt.subplots()
        sns.heatmap(df_tfidf, cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Vector pregunta
        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]

        st.subheader("Resultado")

        st.write(f"**Pregunta:** {question}")
        st.write(f"**Documento más relevante:** Doc {best_idx+1}")

        q_stems = tokenize_and_stem(question)

        highlighted = highlight_words(best_doc, q_stems)

        st.markdown(
            f"**Documento:** {highlighted}",
            unsafe_allow_html=True
        )

        st.write(f"**Similitud:** {similarities[best_idx]:.3f}")

        # 📈 Gráfico de similitud
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Similitud": similarities
        })

        st.subheader("Comparación de similitudes")

        fig2, ax2 = plt.subplots()
        ax2.bar(sim_df["Documento"], sim_df["Similitud"])
        ax2.set_ylabel("Similitud coseno")
        ax2.set_xlabel("Documentos")
        st.pyplot(fig2)

        st.subheader("Ranking de documentos")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))




