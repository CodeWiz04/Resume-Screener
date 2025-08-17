# AI Resume Screener (Streamlit Frontend)

An intelligent Streamlit web app that allows recruiters to upload a resume and paste a job description, then instantly provides a match percentage indicating how well the resume aligns with the job requirements.

📌 Project Overview

- Recruiters often spend hours manually screening resumes. This project automates that process by using Natural Language Processing (NLP) and Machine Learning to:

- Analyze the resume content

- Compare it against the job description

- Provide a match percentage score based on skills, keywords, and semantic similarity

🚀 Features

✔️ User-friendly Streamlit frontend
✔️ Upload resume (PDF, DOCX, TXT)
✔️ Paste job description text
✔️ Generates match percentage score (Excellent / Good / Moderate / Low)
✔️ Visual insights (score visualization, metrics)
✔️ Helps recruiters shortlist candidates faster

🛠️ Tech Stack

- Python 3.9+

- Streamlit → Frontend interface

- NLTK / spaCy → Text preprocessing

- scikit-learn → TF-IDF vectorization & similarity

Sentence Transformers → Semantic similarity

PyMuPDF / python-docx → Resume text extraction
