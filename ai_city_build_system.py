import pandas as pd
import numpy as np
from pathlib import Path

# Constants
PROJECT_STATUSES = ["在建", "续建", "完工", "暂停"]
FUND_TYPES = ["一般债券", "专项债券", "中央综合财力", "土地出让金", "中央预算内投资", "中央补助", "省级补助"]


def generate_projects(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate n simulated project basic info records."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "项目编码": [f"P{100000+i}" for i in range(n)],
        "项目名称": [f"项目{i+1}" for i in range(n)],
        "建设单位": rng.choice(["建设单位A", "建设单位B", "建设单位C"], size=n),
        "项目主管部门": rng.choice(["主管部门1", "主管部门2", "主管部门3"], size=n),
        "立项时间": pd.to_datetime(rng.integers(1577836800, 1609459200, size=n), unit="s"),  # 2020
        "概算批复金额(万元)": rng.integers(500, 20001, size=n),
        "预计开工时间": pd.to_datetime(rng.integers(1609459200, 1640995200, size=n), unit="s"),  # 2021
        "预计完工时间": pd.to_datetime(rng.integers(1640995200, 1672531200, size=n), unit="s"),  # 2022
        "项目状态": rng.choice(PROJECT_STATUSES, size=n)
    })
    return df


def generate_funding(projects: pd.DataFrame, n: int = 1000, seed: int = 24) -> pd.DataFrame:
    """Generate n funding arrangement records linked to projects."""
    rng = np.random.default_rng(seed)
    proj_codes = projects["项目编码"].values
    df = pd.DataFrame({
        "项目编码": rng.choice(proj_codes, size=n),
        "项目名称": None,
        "建设单位": None,
        "项目主管部门": None,
        "概算批复金额(万元)": None,
        "资金批复金额(万元)": rng.integers(100, 5001, size=n),
        "批复日期": pd.to_datetime(rng.integers(1609459200, 1693526400, size=n), unit="s"),  # 2021-2023
        "资金性质": rng.choice(FUND_TYPES, size=n),
        "经办人": rng.choice(["张三", "李四", "王五", "赵六"], size=n),
        "经办处室": rng.choice(["处室A", "处室B"], size=n)
    })
    # Map basic fields from projects table
    base = projects.set_index("项目编码")[["项目名称", "建设单位", "项目主管部门", "概算批复金额(万元)"]]
    df[["项目名称", "建设单位", "项目主管部门", "概算批复金额(万元)"]] = df["项目编码"].map(base.to_dict("index")).apply(pd.Series)
    return df


def export_to_excel(projects: pd.DataFrame, funds: pd.DataFrame, path: str | Path = "ai_city_data.xlsx") -> None:
    """Export two tables to Excel with separate sheets."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        projects.to_excel(writer, sheet_name="项目基本信息表", index=False)
        funds.to_excel(writer, sheet_name="资金安排表", index=False)


def import_from_excel(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    projects = pd.read_excel(path, sheet_name="项目基本信息表")
    funds = pd.read_excel(path, sheet_name="资金安排表")
    return projects, funds


def pivot_funding(funds: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.pivot_table(
        funds,
        index=["项目编码", "项目名称"],
        columns="资金性质",
        values="资金批复金额(万元)",
        aggfunc="sum",
        fill_value=0
    )
    pivot["一般债券"] = pivot.get("一般债券", 0)
    pivot["专项债券"] = pivot.get("专项债券", 0)
    pivot["债券资金合计"] = pivot["一般债券"] + pivot["专项债券"]
    pivot = pivot.reset_index()
    return pivot


def main():
    projects = generate_projects()
    funds = generate_funding(projects)
    export_to_excel(projects, funds)
    pivot = pivot_funding(funds)
    pivot.to_excel("funding_pivot.xlsx", index=False)
    print("Excel files generated: ai_city_data.xlsx and funding_pivot.xlsx")


if __name__ == "__main__":
    main()