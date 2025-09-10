import streamlit as st
from app_utils import initialize_app_state

initialize_app_state()

st.set_page_config(page_title="AI城建系统", page_icon="🏗️", layout="wide")

st.title("AI城建系统")

st.markdown(
	"""
	- 基础数据：维护与导入导出项目基本信息与资金安排。
	- 数据可视化：对资金与项目进行图表分析。
	- 智慧报表生成：按项目与资金性质自动生成数据透视表并导出。
	"""
)

st.page_link("pages/1_基础数据.py", label="基础数据", icon="📚")
st.page_link("pages/2_数据可视化.py", label="数据可视化", icon="📊")
st.page_link("pages/3_智慧报表生成.py", label="智慧报表生成", icon="📑")