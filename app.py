import streamlit as st

st.set_page_config(
    page_title="KrishokBot | কৃষকবট",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

chat_page = st.Page("pages/chat.py", title="Chat", default=True)
about_page = st.Page("pages/about.py", title="About")
limits_page = st.Page("pages/limitations.py", title="Limitations")

nav = st.navigation([chat_page, about_page, limits_page], position="sidebar")
nav.run()
