import streamlit as st
import pandas as pd
import os
import base64
from resume_parser import extract_text_from_pdf
from nlp_model import extract_skills, calculate_similarity

# ===============================
# 🎨 Background Image Setup
# ===============================
def add_bg_from_local(image_file):
    """Adds a background image from a local file"""
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image file (make sure the image is in your main folder)
add_bg_from_local("bg.png")  # Change this to your actual image filename if needed

# ===============================
# 🧠 Streamlit App Starts Here
# ===============================
st.title("📄 AI Resume Ranking System")
st.markdown("### Rank resumes automatically based on the given job description using NLP & AI")

# Create necessary folder if not exists
os.makedirs("data/resumes", exist_ok=True)

# Input area for Job Description
job_desc = st.text_area("🧾 Paste Job Description", height=200, placeholder="Enter the job role and required skills here...")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])

# Process and rank candidates
if st.button("🚀 Rank Candidates"):
    if not uploaded_files:
        st.warning("Please upload at least one resume before ranking.")
    elif not job_desc.strip():
        st.warning("Please enter a job description to compare.")
    else:
        results = []
        with st.spinner("Analyzing resumes... Please wait ⏳"):
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/resumes", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Extract and analyze
                resume_text = extract_text_from_pdf(file_path)
                skills = extract_skills(resume_text)
                score = calculate_similarity(resume_text, job_desc)
                results.append({
                    "Candidate": uploaded_file.name,
                    "Match %": round(score * 100, 2),
                    "Skills Found": ", ".join(skills)
                })

        # Create DataFrame and sort
        df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

        # Display results
        st.success("✅ Ranking completed!")
        st.dataframe(df, use_container_width=True)

        # Save to CSV
        output_path = "data/output.csv"
        df.to_csv(output_path, index=False)
        st.download_button(
            label="📥 Download Ranked Results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_results.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption("Wish you Good Luck for a better future")
