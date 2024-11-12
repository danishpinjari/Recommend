import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV data
user_data = pd.read_csv('user.csv')  # Replace with the actual path to user.csv
job_data = pd.read_csv('job.csv')  # Replace with the actual path to job.csv

# Create a single TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Combine skills from both datasets to fit the vectorizer with the same feature space
combined_skills = pd.concat([user_data['Skills'], job_data['SkillsRequired']], axis=0)

# Fit and transform the combined data once to ensure consistent feature space
tfidf_vectorizer.fit(combined_skills)

# Transform the user data and job data using the same vectorizer
user_tfidf = tfidf_vectorizer.transform(user_data['Skills'])
job_tfidf = tfidf_vectorizer.transform(job_data['SkillsRequired'])

# Streamlit UI
st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("Job Recommendation System")

# Sidebar Styling
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["User Recommendations", "Job Recommendations"])

# Styling options (LinkedIn-inspired color scheme)
st.markdown("""
<style>
    /* General page style */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #054584;  /* LinkedIn light background */
        color: #000;  /* Dark text color for readability */
        padding: 0;
        margin: 0;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #004182;  /* LinkedIn Blue */
        color: #ffffff;
        padding: 20px;
        box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.1);  /* Slight shadow for depth */
    }

    .sidebar .sidebar-content h1 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 20px;
        color: white;
    }

    /* Buttons styling */
    .stButton > button {
        background-color: #0073b1;  /* LinkedIn blue button */
        color: #ffffff;
        padding: 12px 25px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
        font-size: 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #005f8c;  /* Slightly darker blue on hover */
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        padding: 12px;
        border-radius: 4px;
        border: 1px solid #ccc;  /* Subtle border */
        background-color: #ffffff;
        font-size: 1rem;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0073b1;  /* Focused border color matching LinkedIn blue */
        outline: none;
    }

    /* Subheader text style */
    .stSubheader {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333;  /* Dark text for subheaders */
    }

    /* Card styles */
    .card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Soft shadow for cards */
        padding: 20px;
        margin-bottom: 20px;
        font-size: 1rem;
    }

    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #004182;  /* LinkedIn Blue for card headers */
    }

    .card-body {
        color: #333;
        line-height: 1.6;
    }

    /* Selectbox styling */
    .stSelectbox select {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 12px;
        color: #333;
    }

    /* Navbar style */
    .navbar {
        background-color: #ffffff;
        padding: 12px 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Soft shadow for navbar */
        position: sticky;
        top: 0;
        z-index: 10;
    }

    .navbar .navbar-brand {
        color: #0073b1;
        font-weight: 600;
        font-size: 1.5rem;
    }

    /* Link styling */
    a {
        color: #0073b1;  /* LinkedIn blue for links */
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }
</style>

""", unsafe_allow_html=True)

# "User Recommendations" Page
if page == "User Recommendations":
    st.title("User Page - Job Recommendations")
    
    user_input = st.text_area("Enter your skills or job preferences", placeholder="e.g., Python, Data Science, Machine Learning")
    
    if st.button("Get Job Recommendations"):
        if user_input:
            user_vector = tfidf_vectorizer.transform([user_input])
            similarity_scores = cosine_similarity(user_vector, job_tfidf).flatten()
            top_indices = similarity_scores.argsort()[-5:][::-1]

            st.write("### Top job recommendations for you:")
            for index in top_indices:
                st.markdown(f"""
                    <div class="card">
                        <div class="card-header">{job_data.iloc[index]['JobTitle']} at {job_data.iloc[index]['Company']}</div>
                        <div class="card-body">
                            <p><strong>Location:</strong> {job_data.iloc[index]['Location']}</p>
                            <p><strong>Skills Required:</strong> {job_data.iloc[index]['SkillsRequired']}</p>
                            <p><strong>Description:</strong> {job_data.iloc[index]['JobDescription']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter your skills or preferences.")

    # Streamlit form for creating a user profile
    with st.form("user_form"):
        st.header("Create Your Profile")
        
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        location = st.text_input("Location")
        experience = st.text_input("Years of Experience")
        qualification = st.text_input("Qualification")
        skills = st.text_area("Skills (comma-separated)")
        
        submit_button = st.form_submit_button("Submit Profile")
        
        if submit_button:
            # Add the new user profile to the user_data DataFrame
            new_user = pd.DataFrame({
                'Name': [name],
                'Email': [email],
                'Location': [location],
                'Experience': [experience],
                'Qualification': [qualification],
                'Skills': [skills]
            })

            # Append new user profile to the user CSV file
            new_user.to_csv('user.csv', mode='a', header=False, index=False)

            # Reload the updated user data from the CSV
            user_data = pd.read_csv('user.csv')

            # Recalculate the TF-IDF transformation for the updated user data (same as job data)
            combined_skills = pd.concat([user_data['Skills'], job_data['SkillsRequired']], axis=0)
            tfidf_vectorizer.fit(combined_skills)
            user_tfidf = tfidf_vectorizer.transform(user_data['Skills'])
            
            # Display the profile after submission
            st.success("Profile submitted successfully!")
            
            # Recommend jobs based on the new user's skills
            user_vector = tfidf_vectorizer.transform([skills])
            similarity_scores = cosine_similarity(user_vector, job_tfidf).flatten()
            top_indices = similarity_scores.argsort()[-5:][::-1]
            
            st.header("Job Recommendations")
            for index in top_indices:
                st.markdown(f"""
                    <div class="card">
                        <div class="card-header">{job_data.iloc[index]['JobTitle']} at {job_data.iloc[index]['Company']}</div>
                        <div class="card-body">
                            <p><strong>Location:</strong> {job_data.iloc[index]['Location']}</p>
                            <p><strong>Experience Required:</strong> {job_data.iloc[index]['ExperienceRequired']}</p>
                            <p><strong>Qualification Required:</strong> {job_data.iloc[index]['QualificationRequired']}</p>
                            <p><strong>Skills:</strong> {job_data.iloc[index]['SkillsRequired']}</p>
                            <p><strong>Job Description:</strong> {job_data.iloc[index]['JobDescription']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# "Job Recommendations" Page
elif page == "Job Recommendations":
    st.title("Job Page - User Recommendations")
    job_input = st.selectbox("Select a job title", job_data['JobTitle'])

    if st.button("Find Matching Users"):
        if job_input:
            selected_job_index = job_data[job_data['JobTitle'] == job_input].index[0]
            job_vector = job_tfidf[selected_job_index]
            similarity_scores = cosine_similarity(job_vector, user_tfidf).flatten()
            top_indices = similarity_scores.argsort()[-5:][::-1]

            st.write(f"### Top users recommended for the job '{job_input}':")
            for index in top_indices:
                st.markdown(f"""
                    <div class="card">
                        <div class="card-header">{user_data.iloc[index]['Name']}</div>
                        <div class="card-body">
                            <p><strong>Email:</strong> {user_data.iloc[index]['Email']}</p>
                            <p><strong>Location:</strong> {user_data.iloc[index]['Location']}</p>
                            <p><strong>Experience:</strong> {user_data.iloc[index]['Experience']}</p>
                            <p><strong>Skills:</strong> {user_data.iloc[index]['Skills']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please select a job title.")

# Display all existing user profiles from the user CSV
st.header("Existing Profiles")
for _, user in user_data.iterrows():
    st.markdown(f"""
        <div class="card">
            <div class="card-header">{user['Name']}</div>
            <div class="card-body">
                <p><strong>Email:</strong> {user['Email']}</p>
                <p><strong>Location:</strong> {user['Location']}</p>
                <p><strong>Experience:</strong> {user['Experience']}</p>
                <p><strong>Qualification:</strong> {user['Qualification']}</p>
                <p><strong>Skills:</strong> {user['Skills']}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
