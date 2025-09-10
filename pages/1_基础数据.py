import streamlit as st
import pandas as pd
from app_utils import (
	load_projects,
	save_projects,
	load_funding,
	save_funding,
	validate_projects,
	validate_funding,
	to_excel_bytes,
	PROJECT_COLUMNS,
	FUNDING_COLUMNS,
	ALLOWED_PROJECT_STATUS,
	ALLOWED_FUND_NATURE,
	initialize_app_state,
)

st.set_page_config(page_title="基础数据", layout="wide")
initialize_app_state()
st.title("基础数据维护与导入导出")

projects_df = load_projects()
funding_df = load_funding()

with st.expander("项目基本信息表（导入/导出/查看）", expanded=True):
	col_a, col_b, col_c = st.columns([2, 1, 1])
	with col_a:
		st.write("当前项目条数:", len(projects_df))
	with col_b:
		proj_download = st.download_button(
			"导出项目基本信息Excel",
			data=to_excel_bytes(projects_df, sheet_name="项目基本信息"),
			file_name="项目基本信息.xlsx",
		)
	with col_c:
		proj_upload = st.file_uploader("导入项目基本信息Excel", type=["xlsx"], key="proj_upload")
		if proj_upload is not None:
			try:
				df = pd.read_excel(proj_upload)
				ok, msg, df = validate_projects(df)
				if ok:
					save_projects(df)
					st.success("项目基本信息导入成功")
					projects_df = df
				else:
					st.error(msg)
			except Exception as e:
				st.error(f"导入失败: {e}")

	st.dataframe(projects_df, use_container_width=True, hide_index=True)

with st.expander("资金安排表（导入/导出/查看）", expanded=True):
	col_a, col_b, col_c = st.columns([2, 1, 1])
	with col_a:
		st.write("当前资金安排条数:", len(funding_df))
	with col_b:
		fund_download = st.download_button(
			"导出资金安排Excel",
			data=to_excel_bytes(funding_df, sheet_name="资金安排"),
			file_name="资金安排.xlsx",
		)
	with col_c:
		fund_upload = st.file_uploader("导入资金安排Excel", type=["xlsx"], key="fund_upload")
		if fund_upload is not None:
			try:
				df = pd.read_excel(fund_upload)
				ok, msg, df = validate_funding(df, project_codes=projects_df["项目编码"]) if not projects_df.empty else (False, "请先导入项目基本信息表", df)
				if ok:
					save_funding(df)
					st.success("资金安排导入成功")
					funding_df = df
				else:
					st.error(msg)
			except Exception as e:
				st.error(f"导入失败: {e}")

	st.dataframe(funding_df, use_container_width=True, hide_index=True)

with st.expander("字段规范", expanded=False):
	st.write("项目基本信息表字段:")
	st.code("\n".join(PROJECT_COLUMNS))
	st.write("\n项目状态取值: ", ", ".join(ALLOWED_PROJECT_STATUS))
	st.write("\n资金安排表字段:")
	st.code("\n".join(FUNDING_COLUMNS))
	st.write("\n资金性质取值: ", ", ".join(ALLOWED_FUND_NATURE))