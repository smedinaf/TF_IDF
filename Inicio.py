import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.title("AI Document Search with TF-IDF")

st.write("""
This demo shows how **TF-IDF and cosine similarity** can be used to find the most relevant document for a question.

Steps:
1️⃣ Write multiple documents  
2️⃣ Ask a question  
3️⃣ The system finds the most relevant document
""")

# Document input
text_input = st.text_area(
    "Write your documents (one per line):",
    """Artificial intelligence is transforming many industries.
Machine learning allows computers to learn from data.
Neural networks are inspired by the human brain.
Deep learning is widely used in computer vision and NLP."""
)

question = st.text_input(
    "Ask a question:",
    "What technology is used for image recognition?"
)

if st.button("Analyze Documents"):

    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) == 0:
        st.warning("Please enter at least one document.")

    else:

        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.subheader("TF-IDF Matrix")
        st.dataframe(df_tfidf.round(3))

        # Question vector
        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.subheader("Result")

        st.write("**Question:**", question)
        st.write(f"**Most relevant document (Doc {best_idx+1}):**")
        st.success(best_doc)

        st.write(f"Similarity score: **{best_score:.3f}**")

        # Similarity ranking
        sim_df = pd.DataFrame({
            "Document": [f"Doc {i+1}" for i in range(len(documents))],
            "Text": documents,
            "Similarity": similarities
        })

        st.subheader("Similarity Ranking")
        st.dataframe(sim_df.sort_values("Similarity", ascending=False))

        # Mostrar palabras clave importantes
        st.subheader("Top Keywords in Each Document")

        feature_names = vectorizer.get_feature_names_out()

        for i, row in enumerate(X.toarray()):
            top_indices = row.argsort()[-5:][::-1]
            top_words = [feature_names[j] for j in top_indices]

            st.write(f"**Doc {i+1}:** {', '.join(top_words)}")




