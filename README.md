# AI Resume–Job Matcher
A local NLP utility that compares a resume with a job description using TF-IDF and cosine similarity.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Upload a text-based PDF or TXT resume, paste a job description and view the match score plus important missing terms.

## Notes
This is a screening aid, not an actual hiring decision. PDF text extraction works best with selectable text (not scanned images).

## Resume bullet
Built an NLP-based resume matching tool using TF-IDF, cosine similarity and keyword-gap analysis to compare candidate profiles with job descriptions.
