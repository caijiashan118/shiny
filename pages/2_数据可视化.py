import streamlit as st
import pandas as pd
import plotly.express as px
from app_utils import load_projects, load_funding, initialize_app_state

st.set_page_config(page_title="数据可视化", layout="wide")
initialize_app_state()
st.title("数据可视化")

projects_df = load_projects()
funding_df = load_funding()

col1, col2 = st.columns(2)
with col1:
	st.subheader("项目概算金额分布")
	if not projects_df.empty:
		fig = px.histogram(projects_df, x="概算批复金额(万元)", nbins=40)
		st.plotly_chart(fig, use_container_width=True)
	else:
		st.info("暂无项目数据")

with col2:
	st.subheader("项目状态统计")
	if not projects_df.empty:
		status_counts = projects_df["项目状态"].value_counts().reset_index()
		status_counts.columns = ["项目状态", "数量"]
		fig = px.bar(status_counts, x="项目状态", y="数量", text="数量")
		st.plotly_chart(fig, use_container_width=True)
	else:
		st.info("暂无项目数据")

st.subheader("各资金性质年度金额趋势")
if not funding_df.empty:
	funding_df = funding_df.copy()
	funding_df["批复年份"] = pd.to_datetime(funding_df["批复日期"], errors="coerce").dt.year
	trend = funding_df.groupby(["批复年份", "资金性质"], as_index=False)["资金批复金额(万元)"].sum()
	fig = px.line(trend, x="批复年份", y="资金批复金额(万元)", color="资金性质", markers=True)
	st.plotly_chart(fig, use_container_width=True)
else:
	st.info("暂无资金安排数据")