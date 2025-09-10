import streamlit as st
import pandas as pd

from ai_city.io_utils import load_all_tables
from ai_city.pivot import generate_funding_pivot

st.set_page_config(page_title="智慧报表生成", page_icon="🧠", layout="wide")

st.title("智慧报表生成")

projects_df, fundings_df, _ = load_all_tables()

if fundings_df.empty:
    st.info("暂无资金安排数据。请在‘基础数据’中导入或生成模拟数据。")
    st.stop()

pivot_df = generate_funding_pivot(fundings_df, projects_df)

st.subheader("项目资金安排透视表（按资金性质）")
st.dataframe(pivot_df, use_container_width=True, hide_index=True)

@st.cache_data
def _pivot_excel_bytes(df: pd.DataFrame) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="资金安排透视表")
    buf.seek(0)
    return buf.read()

st.download_button(
    label="导出透视表.xlsx",
    data=_pivot_excel_bytes(pivot_df),
    file_name="资金安排透视表.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
