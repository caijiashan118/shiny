from __future__ import annotations

import pandas as pd

from .constants import FUND_NATURE_ALLOWED


def generate_funding_pivot(fundings: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
	if fundings.empty:
		return pd.DataFrame()
	pivot = pd.pivot_table(
		fundings,
		index=["项目编码"],
		columns=["资金性质"],
		values="资金批复金额",
		aggfunc="sum",
		fill_value=0.0,
	)
	for col in FUND_NATURE_ALLOWED:
		if col not in pivot.columns:
			pivot[col] = 0.0
	pivot["一般债券"] = pivot.get("一般债券", 0.0)
	pivot["专项债券"] = pivot.get("专项债券", 0.0)
	pivot["债券资金合计"] = pivot["一般债券"] + pivot["专项债券"]
	pivot["资金合计"] = pivot[FUND_NATURE_ALLOWED].sum(axis=1)
	pivot = pivot.reset_index()
	base = projects[["项目编码", "项目名称", "建设单位", "项目主管部门"]].drop_duplicates()
	result = base.merge(pivot, on="项目编码", how="left").fillna(0.0)
	ordered_cols = [
		"项目编码",
		"项目名称",
		"建设单位",
		"项目主管部门",
		*FUND_NATURE_ALLOWED,
		"债券资金合计",
		"资金合计",
	]
	return result.reindex(columns=ordered_cols)
