import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    h1 {
        font-weight: 700;
        color: #111827;
    }

    h2, h3 {
        color: #1F2937;
    }

    /* Card */
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Section spacing */
    .section {
        margin-top: 30px;
    }

    /* Dropdown */
    .stSelectbox {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


def show_recommendation(model_name):
    st.markdown(f"""
    <div style="
        background-color:#ECFDF5;
        padding:15px;
        border-radius:10px;
        border:1px solid #10B981;
    ">
        <h4 style="color:#065F46;">✅ Recommended Model: {model_name}</h4>
    </div>
    """, unsafe_allow_html=True)