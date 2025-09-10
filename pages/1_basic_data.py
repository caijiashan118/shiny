import streamlit as st
import pandas as pd

from ai_city.constants import (
    PROJECT_COLUMNS,
    FUNDING_COLUMNS,
    PROJECT_STATUS_ALLOWED,
    FUND_NATURE_ALLOWED,
)
from ai_city.io_utils import (
    load_all_tables,
    export_tables_to_excel,
    export_template_excel,
    save_projects,
    save_fundings,
    save_key_projects,
    validate_linkage,
)

st.set_page_config(page_title="基础数据", page_icon="📚", layout="wide")

st.title("基础数据")

projects_df, fundings_df, key_projects_df = load_all_tables()

tab1, tab2, tab3 = st.tabs(["项目基本信息表", "资金安排表", "重点项目清单"])

with tab1:
    st.subheader("项目基本信息表")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("导出Excel模板"):
            st.download_button(
                label="下载模板.xlsx",
                data=export_template_excel(),
                file_name="模板_项目基本信息_资金安排.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with colB:
        upload = st.file_uploader("导入Excel（包含‘项目基本信息’表）", type=["xlsx"], key="proj_upload")
        if upload is not None:
            from ai_city.io_utils import import_excel
            proj_in, _ = import_excel(upload.getvalue())
            if not proj_in.empty:
                save_projects(proj_in)
                projects_df = proj_in
                st.success("项目基本信息表已导入并保存")
    with colC:
        if not projects_df.empty:
            st.download_button(
                label="导出当前项目数据",
                data=export_tables_to_excel(projects_df, pd.DataFrame(columns=FUNDING_COLUMNS)),
                file_name="项目基本信息.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.dataframe(projects_df, use_container_width=True, hide_index=True)
    st.caption("项目状态允许：" + ", ".join(PROJECT_STATUS_ALLOWED))

with tab2:
    st.subheader("资金安排表")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("导出Excel模板", key="fund_tpl"):
            st.download_button(
                label="下载模板.xlsx",
                data=export_template_excel(),
                file_name="模板_项目基本信息_资金安排.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with colB:
        upload2 = st.file_uploader("导入Excel（包含‘资金安排’表）", type=["xlsx"], key="fund_upload")
        if upload2 is not None:
            from ai_city.io_utils import import_excel
            _, funds_in = import_excel(upload2.getvalue())
            if not funds_in.empty:
                missing = validate_linkage(funds_in, projects_df)
                if not missing.empty:
                    st.warning(f"有 {len(missing)} 条资金安排的项目编码未在项目基本信息中找到，这些记录将被忽略保存。")
                    funds_in = funds_in[~funds_in.index.isin(missing.index)]
                save_fundings(funds_in)
                fundings_df = funds_in
                st.success("资金安排表已导入并保存")
    with colC:
        if not fundings_df.empty:
            st.download_button(
                label="导出当前资金安排数据",
                data=export_tables_to_excel(pd.DataFrame(columns=PROJECT_COLUMNS), fundings_df),
                file_name="资金安排.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.dataframe(fundings_df, use_container_width=True, hide_index=True)
    st.caption("资金性质允许：" + ", ".join(FUND_NATURE_ALLOWED))

    if not fundings_df.empty and not projects_df.empty:
        missing = validate_linkage(fundings_df, projects_df)
        if missing.empty:
            st.success("资金安排表中的项目编码已与项目基本信息表全部联通。")
        else:
            st.error(f"未联通记录：{len(missing)}。请检查并修正项目编码。")

with tab3:
    st.subheader("重点项目清单")
    if not projects_df.empty:
        selected = st.multiselect(
            "选择重点项目（按项目编码）",
            options=projects_df["项目编码"].tolist(),
            default=key_projects_df["项目编码"].tolist() if not key_projects_df.empty else [],
        )
        if st.button("保存重点项目"):
            to_save = projects_df[projects_df["项目编码"].isin(selected)][["项目编码", "项目名称"]]
            save_key_projects(to_save)
            st.success("重点项目清单已保存")
    st.dataframe(key_projects_df, use_container_width=True, hide_index=True)
