import streamlit as st

from utils.main import check_password

st.set_page_config(
    page_title="ReadMe",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.vizvision.com/',
        'Report a bug': "https://www.vizvision.com/",
        'About': "这是一个视觉算法工具集。"
    }
)

if check_password():
    st.markdown("### 支持的任务: ")
    st.markdown("- :blue[目标检测]")
