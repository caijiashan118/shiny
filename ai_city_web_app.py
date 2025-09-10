import streamlit as st
import pandas as pd
from pathlib import Path
from ai_city_build_system import (
    generate_projects,
    generate_funding,
    export_to_excel,
    import_from_excel,
    pivot_funding,
)

st.set_page_config(page_title="AI城建系统", layout="wide")

DATA_PATH = Path("ai_city_data.xlsx")
PIVOT_PATH = Path("funding_pivot.xlsx")

# Sidebar actions
with st.sidebar:
    st.header("数据操作")
    if st.button("生成模拟数据"):
        projects = generate_projects()
        funds = generate_funding(projects)
        export_to_excel(projects, funds, DATA_PATH)
        pivot = pivot_funding(funds)
        pivot.to_excel(PIVOT_PATH, index=False)
        st.success("已生成模拟数据及报表")
    uploaded = st.file_uploader("导入Excel", type=["xlsx"])
    if uploaded is not None:
        DATA_PATH.write_bytes(uploaded.read())
        st.success("已上传Excel 文件")

# Load data if exists
projects_df = funds_df = pivot_df = None
if DATA_PATH.exists():
    projects_df, funds_df = import_from_excel(DATA_PATH)
if PIVOT_PATH.exists():
    pivot_df = pd.read_excel(PIVOT_PATH)

# Tabs
tabs = st.tabs(["基础数据", "数据可视化", "智慧报表生成"])

with tabs[0]:
    st.subheader("项目基本信息表")
    if projects_df is not None:
        st.dataframe(projects_df)
    else:
        st.info("暂无数据，请先生成或上传数据")

    st.subheader("资金安排表")
    if funds_df is not None:
        st.dataframe(funds_df)
    else:
        st.info("暂无数据")

with tabs[1]:
    st.subheader("项目资金组成可视化")
    if pivot_df is not None:
        import plotly.express as px
        fund_cols = [c for c in pivot_df.columns if c not in ["项目编码", "项目名称"]]
        melted = pivot_df.melt(id_vars=["项目编码", "项目名称"], value_vars=fund_cols, var_name="资金性质", value_name="金额")
        fig = px.bar(melted, x="项目名称", y="金额", color="资金性质", title="各项目资金来源对比", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无数据")

with tabs[2]:
    st.subheader("智慧报表-资金安排透视")
    if pivot_df is not None:
        st.dataframe(pivot_df)
        st.download_button("下载报表", data=PIVOT_PATH.read_bytes(), file_name="funding_pivot.xlsx")
    else:
        st.info("暂无数据")

st.caption("© 2025 AI 城建系统")