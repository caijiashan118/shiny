import streamlit as st
import pandas as pd
import altair as alt

from ai_city.io_utils import load_all_tables

st.set_page_config(page_title="数据可视化", page_icon="📊", layout="wide")

st.title("数据可视化")

projects_df, fundings_df, key_projects_df = load_all_tables()

if fundings_df.empty:
    st.info("暂无资金安排数据。请在‘基础数据’中导入或生成模拟数据。")
    st.stop()

nature_sum = fundings_df.groupby("资金性质", as_index=False)["资金批复金额"].sum()
chart1 = (
    alt.Chart(nature_sum)
    .mark_bar()
    .encode(x=alt.X("资金性质:N", sort="-y"), y="资金批复金额:Q", tooltip=["资金性质", "资金批复金额"])
    .properties(height=350)
)

project_total = fundings_df.groupby(["项目编码", "项目名称"], as_index=False)["资金批复金额"].sum()
project_total = project_total.sort_values("资金批复金额", ascending=False).head(20)
chart2 = (
    alt.Chart(project_total)
    .mark_bar()
    .encode(x=alt.X("资金批复金额:Q"), y=alt.Y("项目名称:N", sort="-x"), tooltip=["项目名称", "资金批复金额"])
    .properties(height=600)
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("按资金性质合计")
    st.altair_chart(chart1, use_container_width=True)
with col2:
    st.subheader("项目资金合计 Top 20")
    st.altair_chart(chart2, use_container_width=True)

if not key_projects_df.empty:
    st.subheader("重点项目资金对比")
    kp = fundings_df.merge(key_projects_df, on=["项目编码", "项目名称"], how="inner")
    kp_sum = kp.groupby("项目名称", as_index=False)["资金批复金额"].sum().sort_values("资金批复金额", ascending=False)
    chart3 = (
        alt.Chart(kp_sum)
        .mark_bar(color="#f39c12")
        .encode(x=alt.X("资金批复金额:Q"), y=alt.Y("项目名称:N", sort="-x"), tooltip=["项目名称", "资金批复金额"])
        .properties(height=400)
    )
    st.altair_chart(chart3, use_container_width=True)
