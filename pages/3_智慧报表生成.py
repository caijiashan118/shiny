import streamlit as st
import pandas as pd
from app_utils import load_funding, build_pivot_table, to_excel_bytes, initialize_app_state

st.set_page_config(page_title="智慧报表生成", layout="wide")
initialize_app_state()
st.title("智慧报表生成（数据透视表）")

funding_df = load_funding()

if funding_df.empty:
	st.info("请先在基础数据中导入或生成资金安排数据")
	st.stop()

st.caption("以项目编码为唯一识别信息，汇总不同资金性质的资金安排。")
pivot_df = build_pivot_table(funding_df)

st.dataframe(pivot_df, use_container_width=True, hide_index=True)

st.download_button(
	"导出数据透视表",
	data=to_excel_bytes(pivot_df, sheet_name="数据透视表"),
	file_name="项目资金透视表.xlsx",
)