import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import re

st.set_page_config(page_title="Resume–Job Matcher", page_icon="🎯")
st.title("🎯 AI Resume–Job Matcher")
st.write("Compare a resume with a job description using lightweight NLP.")
file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
jd = st.text_area("Paste the job description", height=220)

def extract(upload):
    if upload.name.lower().endswith(".txt"):
        return upload.getvalue().decode("utf-8", errors="ignore")
    reader = PdfReader(upload)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def words(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z+#.]{1,}", text.lower()))

if st.button("Analyze match", type="primary"):
    if not file or not jd.strip():
        st.warning("Upload a resume and paste a job description.")
    else:
        resume = extract(file)
        if not resume.strip():
            st.error("No selectable text found. Try a text-based PDF or TXT file.")
        else:
            matrix = TfidfVectorizer(stop_words="english").fit_transform([resume, jd])
            score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)
            missing = sorted(words(jd) - words(resume))
            st.metric("Text similarity score", f"{score:.1f}%")
            st.progress(min(int(score), 100))
            st.subheader("Potential keyword gaps")
            st.write(", ".join(missing[:35]) if missing else "No obvious keyword gaps.")
            st.caption("Similarity is a rough signal. Review skills, experience and context manually.")
