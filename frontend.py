import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# --- Load model and vectorizers ---
model = joblib.load("logreg_tfidf_model.joblib")
tfidf_text = joblib.load("tfidf_text_vectorizer.joblib")
tfidf_gene = joblib.load("tfidf_gene_vectorizer.joblib")
tfidf_var = joblib.load("tfidf_var_vectorizer.joblib")

# --- Preprocessing function ---
stop_words = set(stopwords.words('english'))
def nlp_preprocessing(text):
    if type(text) is not int:
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([word for word in text.split() if word not in stop_words])
    else:
        return ""

# --- Streamlit UI ---
st.set_page_config(page_title="Personalised Cancer Diagnosis", page_icon="🧬", layout="centered")

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%); padding: 2rem 1rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h1 style='text-align: center; color: white; margin-bottom: 0.2em;'>🧬 Personalised Cancer Diagnosis</h1>
        <h3 style='text-align: center; color: #e0f7fa;'>AI-powered Genetic Variant Classifier</h3>
    </div>
    """, unsafe_allow_html=True
)

with st.expander("ℹ️ About & Instructions", expanded=True):
    st.markdown("""
**Project Motivation:**  
This project was built for the [MSK - Redefining Cancer Treatment](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/overview) competition, organized by Memorial Sloan Kettering Cancer Center (MSK), a world leader in cancer research and care.

**Purpose:**  
To help clinicians and researchers predict the class of a genetic variant (mutation) based on gene, mutation/variation, and clinical text, supporting more personalized cancer diagnosis and treatment.

**How to use:**  
1. **Enter the gene name** (e.g., `BRCA1`).
2. **Enter the variation/mutation** (e.g., `A1699S`).
3. **(Optional)** Add any clinical text or description.
4. Click **Classify Variant** to see the predicted class and interpretation.

**What is a gene/variation?**  
- A **gene** is a segment of DNA that codes for a protein.
- A **variation** (mutation) is a change in the DNA sequence of a gene, which may affect how the gene works.
- **Clinical text** can provide extra context from lab reports or studies.

**Who is this for?**  
Doctors, researchers, and students interested in cancer genomics and AI-powered diagnostics.

**Reference:**  
NIPS 2017 Competition Track  
**Author:** [Aditya Singh](https://www.linkedin.com/in/aditya-singh-2b319b299/) | [GitHub](https://github.com/AdiSinghCodes)
    """)

st.markdown("---")

st.markdown("### 📝 Enter Variant Details")
gene = st.text_input("🧬 Gene Name", value="", help="e.g. BRCA1, TP53, EGFR")
variation = st.text_input("🔬 Variation / Mutation", value="", help="e.g. A1699S, R273H, Deletion")
text = st.text_area("📄 Clinical Text / Description (Optional)", value="", help="Any relevant clinical or research description.")

run_button = st.button("🚀 Classify Variant", use_container_width=True)

if run_button:
    if not gene.strip() or not variation.strip():
        st.warning("Please enter both the gene name and variation/mutation to proceed.", icon="⚠️")
    else:
        # --- Preprocess and vectorize ---
        gene_proc = gene
        var_proc = variation
        text_proc = nlp_preprocessing(text)

        gene_feat = tfidf_gene.transform([gene_proc])
        var_feat = tfidf_var.transform([var_proc])
        text_feat = tfidf_text.transform([text_proc])
        X = np.hstack([gene_feat.toarray(), var_feat.toarray(), text_feat.toarray()])

        # --- Prediction ---
        pred_class = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        class_labels = [f"Class {i}" for i in range(1, len(proba)+1)]

        # --- Feature presence ---
        gene_vocab = tfidf_gene.get_feature_names_out()
        var_vocab = tfidf_var.get_feature_names_out()
        text_vocab = tfidf_text.get_feature_names_out()

        present_genes = [g for g in gene_vocab if g in gene_proc.lower()]
        present_vars = [v for v in var_vocab if v in var_proc.lower()]
        present_words = [w for w in text_vocab if w in text_proc.split()]

        # --- Interpretability: Top features for predicted class ---
        coefs = model.estimator.coef_[pred_class-1]
        feature_names = np.concatenate([gene_vocab, var_vocab, text_vocab])
        top_indices = np.argsort(coefs)[-10:][::-1]
        top_features = [(feature_names[idx], coefs[idx]) for idx in top_indices]

        # --- Results ---
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%); padding: 1.2rem 1rem; border-radius: 10px; margin-bottom: 1.2rem;">
                <h2 style='color:#005bea; margin-bottom:0.2em;'>🎯 Predicted Class: <span style='color:#00bfff;'>Class {pred_class}</span></h2>
                <p style='font-size:1.1em; color:#333;'>Confidence: <b>{100*proba[pred_class-1]:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("#### Class Probabilities")
        st.bar_chart(pd.Series(proba, index=class_labels))

        st.markdown("### 🔍 Feature Analysis")
        st.markdown(f"- **Gene:** `{gene}` {'✅' if present_genes else '❌'}")
        st.markdown(f"- **Variation:** `{variation}` {'✅' if present_vars else '❌'}")
        st.markdown(f"- **Text features present:** {', '.join(present_words) if present_words else 'None'}")

        st.markdown("#### Top Features Contributing to This Prediction")
        for fname, coef in top_features:
            st.write(f"- `{fname}`: {coef:.4f}")

        st.info("**Interpretability:** All features above contributed to the model's decision. Please review them to assess the prediction's reliability.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:0.95em;">
        Made with ❤️ by <a href="https://www.linkedin.com/in/aditya-singh-2b319b299/" target="_blank">Aditya Singh</a> | 
        <a href="https://github.com/AdiSinghCodes" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True

)
