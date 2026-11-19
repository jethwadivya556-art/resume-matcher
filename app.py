import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_pdf(file):
    """Extract text from an uploaded PDF."""
    reader = PdfReader(file)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def clean(text):
    """Lowercase and remove symbols so words compare fairly."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def match_score(resume, jd):
    """TF-IDF + cosine similarity between resume and job description (0-100)."""
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([clean(resume), clean(jd)])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0]) * 100


def missing_keywords(resume, jd, top_n=25):
    """Important words in the job description that are not in the resume."""
    vec = TfidfVectorizer(stop_words="english")
    scores = vec.fit_transform([clean(jd)]).toarray()[0]
    terms = vec.get_feature_names_out()
    ranked = [t for _, t in sorted(zip(scores, terms), reverse=True)][:top_n]
    resume_words = set(clean(resume).split())
    return [t for t in ranked if t not in resume_words]


st.set_page_config(page_title="Resume Matcher", page_icon="📄")
st.title("📄 Resume vs Job Description Matcher")
st.write("Upload your resume, paste a job description, and see how well they match.")

uploaded = st.file_uploader("Upload your resume (PDF)", type="pdf")
jd_text = st.text_area("Paste the job description here", height=250)

if st.button("Check match"):
    if not uploaded or not jd_text.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        resume_text = read_pdf(uploaded)
        if not resume_text.strip():
            st.error("Could not read text from this PDF. Try a text-based PDF.")
        else:
            score = match_score(resume_text, jd_text)
            st.metric("Match score", f"{score:.1f}%")
            st.progress(min(int(score), 100))

            missing = missing_keywords(resume_text, jd_text)
            if missing:
                st.subheader("Keywords in the job description missing from your resume")
                st.write(", ".join(missing))
                st.caption("Add these only if you genuinely have the skill.")
            else:
                st.success("Your resume covers the main keywords!")
