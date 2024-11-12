# Job Recommendation System using Streamlit

## Overview

This project is a Job Recommendation System built using Streamlit, Python, and Scikit-learn. It provides users with job recommendations based on their skills and preferences, and vice versa. The system also allows users to create their profile and receive job suggestions based on their skillset. Additionally, it recommends users for a particular job based on their skills.

## Features

- **User Recommendations**: Input a list of skills, and the system will recommend jobs that match those skills.
- **Job Recommendations**: Select a job title, and the system will suggest users with matching skills for that job.
- **Profile Creation**: Users can create and submit their profiles, which will then be used to provide tailored job recommendations.
- **Data Storage**: User profiles and job data are stored in CSV files (`user.csv` and `job.csv`).
- **TF-IDF Vectorization**: The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the skills and job descriptions to compute the similarity between users and jobs using cosine similarity.

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For implementing the TF-IDF vectorizer and calculating cosine similarity.
- **Pandas**: For data manipulation and storage (CSV files).
- **HTML/CSS**: For custom styling the Streamlit interface (LinkedIn-inspired theme).

## Installation

### Prerequisites

- Python 3.6+
- Streamlit
- Pandas
- Scikit-learn

### Install Dependencies

```bash
pip install streamlit pandas scikit-learn
