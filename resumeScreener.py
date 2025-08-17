import streamlit as st
import os
import fitz  # PyMuPDF for PDF
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import time
import nltk

# Page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide"
)

# Sidebar theme toggle
theme = st.sidebar.radio("🎨 Theme Mode", ["Light", "Dark"], index=0)

# Dynamic CSS
if theme == "Light":
    primary, secondary, accent, success, danger, bg = "#4f46e5", "#14b8a6", "#f59e0b", "#22c55e", "#ef4444", "#f9fafb"
    text_color = "#333"
else:
    primary, secondary, accent, success, danger, bg = "#6366f1", "#2dd4bf", "#fbbf24", "#4ade80", "#f87171", "#111827"
    text_color = "#f3f4f6"

st.markdown(f"""
<style>
:root {{
    --primary: {primary};
    --secondary: {secondary};
    --accent: {accent};
    --success: {success};
    --danger: {danger};
    --light-bg: {bg};
    --text-color: {text_color};
}}

body {{
    background: var(--light-bg);
    color: var(--text-color);
}}

/* Header */
.header-container {{
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}}

/* Score containers */
.score-excellent {{ background: linear-gradient(135deg, var(--success), #86efac); }}
.score-good      {{ background: linear-gradient(135deg, var(--secondary), #5eead4); }}
.score-moderate  {{ background: linear-gradient(135deg, var(--accent), #fde68a); color: #333; }}
.score-low       {{ background: linear-gradient(135deg, var(--danger), #fecaca); color: #fff; }}

.score-text {{
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center

}}
.score-description {{
    font-size: 1rem;
    opacity: 0.9;
    text-align: center;
    margin-top: 0.5rem
}}

/* Upload & textarea alignment */
.stFileUploader, .stTextArea {{ height: 260px !important; }}
.stFileUploader > div > div > div {{
    min-height: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}}

/* Metric cards */
.metric-card {{
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    height: 150px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    border-left: 4px solid var(--primary);
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.metric-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
}}
.metric-label {{
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.5rem;
}}

/* Button */
.stButton > button {{
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border: none !important;
    padding: 0.9rem 2rem !important;
    border-radius: 25px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 5px 15px rgba(79,70,229,0.3);
    transition: all 0.3s ease;
}}
.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 7px 20px rgba(79,70,229,0.4);
}}
</style>
""", unsafe_allow_html=True)

# Download required NLTK data (run once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# ---------- FILE READING FUNCTIONS ----------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# ---------- ENHANCED TEXT PROCESSING ----------
def clean_and_normalize_text(text):
    """Enhanced text cleaning and normalization"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_keywords(text, max_features=100):
    """Extract important keywords using TF-IDF"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
        
        processed_text = ' '.join(tokens)
        
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw[0] for kw in keyword_scores[:max_features]]
    except:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(words))

def calculate_keyword_overlap(resume_keywords, job_keywords):
    """Calculate keyword overlap percentage"""
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    if not job_set:
        return 0, set()
    
    overlap = resume_set.intersection(job_set)
    overlap_percentage = (len(overlap) / len(job_set)) * 100
    
    return overlap_percentage, overlap

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks for better semantic analysis"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def calculate_semantic_similarity(resume_text, job_desc):
    """Calculate semantic similarity using sentence transformers"""
    model = load_sentence_transformer()
    
    resume_chunks = chunk_text(resume_text, 200)
    
    resume_embeddings = model.encode(resume_chunks)
    job_embedding = model.encode([job_desc])
    
    similarities = cosine_similarity(resume_embeddings, job_embedding)
    max_similarity = np.max(similarities)
    avg_similarity = np.mean(similarities)
    
    return max_similarity, avg_similarity

def calculate_tfidf_similarity(resume_text, job_desc):
    """Calculate TF-IDF based similarity"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0

def calculate_composite_score(semantic_max, semantic_avg, tfidf_sim, keyword_overlap):
    """Calculate a composite matching score"""
    composite_score = (
        semantic_max * 0.3 +
        semantic_avg * 0.2 +
        tfidf_sim * 0.3 +
        (keyword_overlap / 100) * 0.2
    )
    return composite_score * 100

def create_score_visualization(scores_dict):
    """Create a radar chart for score visualization"""
    fig = go.Figure()
    
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgb(102, 126, 234)', width=3),
        name='Match Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Match Analysis Breakdown",
        title_x=0.5,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ---------- STREAMLIT UI ----------

# Custom header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎯 AI Resume Screener</h1>
    <p class="header-subtitle">Advanced AI-powered resume analysis with comprehensive matching algorithms</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### 🔧 Analysis Settings")
    
    analysis_type = st.selectbox(
        "Choose Analysis Method:",
        ["Composite Score (Recommended)", "Semantic Only", "Keyword Only", "TF-IDF Only"],
        help="Select the analysis method that best suits your needs"
    )
    
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    show_visualization = st.checkbox("Show Score Visualization", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Score Interpretation")
    st.markdown("""
    - **🟢 70-100%**: Excellent match
    - **🟡 50-69%**: Good match  
    - **🟠 30-49%**: Moderate match
    - **🔴 0-29%**: Low match
    """)

# Main content area with perfectly aligned boxes
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 Upload Resume")
    fileName = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
        label_visibility="collapsed"
    )
    
    # Show upload status below the upload area
    if fileName is not None:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        ">
            ✅ <strong>{fileName.name}</strong> uploaded successfully!<br>
            <small>File size: {fileName.size / 1024:.1f} KB</small>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 💼 Job Description")
    job_desc = st.text_area(
        "Paste the job description here:",
        height=240,
        placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...",
        help="Include the complete job posting for better analysis",
        label_visibility="collapsed"
    )

# Analysis button with custom styling
st.markdown("<br>", unsafe_allow_html=True)
col_center = st.columns([1, 2, 1])[1]
with col_center:
    analyze_button = st.button(
        "🚀 Analyze Resume Match",
        type="primary",
        use_container_width=True,
        help="Click to start comprehensive resume analysis"
    )

if analyze_button:
    if fileName is None or not job_desc.strip():
        st.error("⚠️ Please upload a resume and enter a job description.")
    else:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing resume and job description..."):
            # Update progress
            status_text.text("📄 Extracting text from resume...")
            progress_bar.progress(20)
            
            # Extract text
            _, ext = os.path.splitext(fileName.name)
            ext = ext.lower()

            if ext == ".pdf":
                resume_text = extract_text_from_pdf(fileName)
            elif ext == ".docx":
                resume_text = extract_text_from_docx(fileName)
            elif ext == ".txt":
                resume_text = extract_text_from_txt(fileName)
            else:
                st.error("❌ Unsupported file type.")
                st.stop()

            # Clean texts
            status_text.text("🧹 Cleaning and preprocessing text...")
            progress_bar.progress(40)
            
            resume_clean = clean_and_normalize_text(resume_text)
            job_clean = clean_and_normalize_text(job_desc)
            
            if not resume_clean or not job_clean:
                st.error("❌ Could not extract text from the uploaded file or job description is empty.")
                st.stop()

            # Perform analysis
            status_text.text("🤖 Running AI analysis...")
            progress_bar.progress(60)
            
            # Initialize variables
            semantic_max = semantic_avg = tfidf_similarity = keyword_overlap_pct = 0
            overlapping_keywords = set()
            
            if analysis_type in ["Composite Score (Recommended)", "Semantic Only"]:
                semantic_max, semantic_avg = calculate_semantic_similarity(resume_clean, job_clean)
                
            if analysis_type in ["Composite Score (Recommended)", "TF-IDF Only"]:
                tfidf_similarity = calculate_tfidf_similarity(resume_clean, job_clean)
                
            if analysis_type in ["Composite Score (Recommended)", "Keyword Only"]:
                resume_keywords = extract_keywords(resume_clean)
                job_keywords = extract_keywords(job_clean)
                keyword_overlap_pct, overlapping_keywords = calculate_keyword_overlap(resume_keywords, job_keywords)

            status_text.text("📊 Calculating final scores...")
            progress_bar.progress(80)

            # Calculate final score
            if analysis_type == "Composite Score (Recommended)":
                final_score = calculate_composite_score(semantic_max, semantic_avg, tfidf_similarity, keyword_overlap_pct)
                score_type = "Composite"
            elif analysis_type == "Semantic Only":
                final_score = semantic_max * 100
                score_type = "Semantic"
            elif analysis_type == "Keyword Only":
                final_score = keyword_overlap_pct
                score_type = "Keyword Overlap"
            elif analysis_type == "TF-IDF Only":
                final_score = tfidf_similarity * 100
                score_type = "TF-IDF"

            final_score = min(round(final_score, 2), 100)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        # Display results with custom styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine score class for styling
        if final_score >= 70:
            score_class = "score-excellent"
            emoji = "🎉"
            message = "Excellent match! Your resume aligns perfectly with the job requirements."
        elif final_score >= 50:
            score_class = "score-good"
            emoji = "✅"
            message = "Good match with room for improvement. Consider adding more relevant keywords."
        elif final_score >= 30:
            score_class = "score-moderate"
            emoji = "⚠️"
            message = "Moderate match. Your resume could benefit from significant tailoring."
        else:
            score_class = "score-low"
            emoji = "🔄"
            message = "Low match. Consider restructuring your resume to better align with this job."

        # Main score display
        st.markdown(f"""
        <div class="score-container {score_class}">
            <p class="score-text">{emoji} {final_score}%</p>
            <p class="score-description">{score_type} Match Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"📝 **Analysis Result:** {message}")

        # Detailed analysis section
        if show_details:
            st.markdown("---")
            st.markdown("## 📊 Detailed Analysis")
            
            # Create metrics in a beautiful layout
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{semantic_max*100:.1f}%</p>
                    <p class="metric-label">Best Semantic Match</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{semantic_avg*100:.1f}%</p>
                    <p class="metric-label">Avg Semantic Match</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{tfidf_similarity*100:.1f}%</p>
                    <p class="metric-label">TF-IDF Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{keyword_overlap_pct:.1f}%</p>
                    <p class="metric-label">Keyword Overlap</p>
                </div>
                """, unsafe_allow_html=True)

            # Score visualization
            if show_visualization and analysis_type == "Composite Score (Recommended)":
                st.markdown("### 📈 Score Breakdown")
                scores_dict = {
                    "Semantic Max": semantic_max * 100,
                    "Semantic Avg": semantic_avg * 100,
                    "TF-IDF": tfidf_similarity * 100,
                    "Keywords": keyword_overlap_pct
                }
                fig = create_score_visualization(scores_dict)
                st.plotly_chart(fig, use_container_width=True)

            # Keywords analysis
            # if overlapping_keywords:
            #     st.markdown("### 🔑 Matching Keywords")
            #     st.markdown("**Keywords found in both resume and job description:**")
            #     keywords_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in sorted(list(overlapping_keywords))])
            #     st.markdown(f'<div style="margin: 1rem 0;">{keywords_html}</div>', unsafe_allow_html=True)
            
            # # Missing keywords
            # if analysis_type in ["Composite Score (Recommended)", "Keyword Only"]:
            #     missing_keywords = set(job_keywords) - set(resume_keywords)
            #     if missing_keywords:
            #         st.markdown("### 📝 Suggested Keywords to Add")
            #         st.markdown("**Important job keywords missing from your resume:**")
            #         missing_html = "".join([f'<span class="missing-keyword-tag">{kw}</span>' for kw in sorted(list(missing_keywords))[:15]])
            #         st.markdown(f'<div style="margin: 1rem 0;">{missing_html}</div>', unsafe_allow_html=True)

# # Enhanced instructions
# st.markdown("---")
# with st.expander("ℹ️ How to Use This Advanced Resume Screener", expanded=False):
#     st.markdown("""
#     ### 🚀 Getting Started
    
#     **Step-by-step Guide:**
#     1. **📄 Upload Resume:** Choose your resume file (PDF, DOCX, or TXT format)
#     2. **💼 Add Job Description:** Copy and paste the complete job posting
#     3. **⚙️ Select Analysis Method:** Choose from the sidebar options
#     4. **🔍 Run Analysis:** Click the "Analyze Resume Match" button
#     5. **📊 Review Results:** Get comprehensive matching insights
    
#     ### 🧠 Analysis Methods Explained
    
#     - **🏆 Composite Score (Recommended):** 
#       - Combines all techniques for most accurate results
#       - Uses weighted scoring: 30% semantic + 30% TF-IDF + 20% keywords + 20% context
    
#     - **🤖 Semantic Only:** 
#       - AI-powered understanding of meaning and context
#       - Best for matching conceptual similarities
    
#     - **🔤 Keyword Only:** 
#       - Direct keyword matching analysis
#       - Perfect for ATS system simulation
    
#     - **📈 TF-IDF Only:** 
#       - Traditional text similarity algorithms
#       - Good for document-level comparison
    
#     ### 💡 Pro Tips for Better Matches
    
#     - **Use Job Keywords:** Include exact terms from the job posting
#     - **Match Job Structure:** Organize your resume similarly to job requirements
#     - **Quantify Achievements:** Use numbers and metrics where possible
#     - **Tailor Experience:** Adjust descriptions to match job responsibilities
#     - **Industry Terms:** Include relevant industry-specific terminology
    
#     ### 🎯 Score Interpretation
    
#     - **90-100%:** Perfect match - submit with confidence
#     - **80-89%:** Excellent match - minor tweaks recommended  
#     - **70-79%:** Very good match - few adjustments needed
#     - **60-69%:** Good match - moderate tailoring required
#     - **50-59%:** Fair match - significant improvements needed
#     - **Below 50%:** Poor match - major restructuring recommended
#     """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Made with ❤️ using Streamlit • Advanced AI Resume Analysis • Version 2.0</p>
</div>
""", unsafe_allow_html=True)