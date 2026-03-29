import spacy

nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
    # Example skill list
    skill_keywords = ["python", "java", "machine learning", "nlp", "flask", "streamlit", "sql"]
    found_skills = [skill for skill in skill_keywords if skill in tokens]
    return list(set(found_skills))


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(resume_text, job_desc):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_emb = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.cos_sim(resume_emb, job_emb)
    return float(similarity[0][0])
