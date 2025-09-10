import streamlit as st
from ai_city.io_utils import ensure_data_storage, load_all_tables, save_projects, save_fundings, save_key_projects
from ai_city.simulate import simulate_projects, simulate_fundings

st.set_page_config(page_title="AI城建系统", page_icon="🏗️", layout="wide")

st.title("AI城建系统")
st.caption("基础数据 · 数据可视化 · 智慧报表生成")

ensure_data_storage()

with st.sidebar:
	st.header("操作")
	if st.button("生成模拟数据", use_container_width=True):
		projects_df = simulate_projects(200)
		save_projects(projects_df)
		fundings_df = simulate_fundings(projects_df, 1000)
		save_fundings(fundings_df)
		# 生成一个简单的重点项目清单（随机选择若干项目）
		key_projects = projects_df.sample(n=min(30, len(projects_df)), random_state=42)[["项目编码", "项目名称"]]
		save_key_projects(key_projects)
		st.success("已生成并保存：200个项目与1000条资金安排，以及重点项目清单")

st.info("使用左侧 Pages 导航进入：基础数据、数据可视化、智慧报表生成 模块。")

# 展示当前数据概览
projects, fundings, key_projects = load_all_tables()
col1, col2, col3 = st.columns(3)
with col1:
	st.metric("项目数量", len(projects))
with col2:
	st.metric("资金安排条数", len(fundings))
with col3:
	st.metric("重点项目数量", len(key_projects))

st.dataframe(projects.head(20), use_container_width=True, hide_index=True)
