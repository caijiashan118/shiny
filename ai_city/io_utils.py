from __future__ import annotations

import io
import os
from typing import Tuple

import pandas as pd

from .constants import (
	DATA_DIR,
	FUNDING_COLUMNS,
	FUNDINGS_CSV,
	KEY_PROJECTS_CSV,
	KEY_PROJECT_COLUMNS,
	PROJECTS_CSV,
	PROJECT_COLUMNS,
	PROJECT_STATUS_ALLOWED,
	FUND_NATURE_ALLOWED,
)


def ensure_data_storage() -> None:
	os.makedirs(DATA_DIR, exist_ok=True)


def _empty_df(columns: list[str]) -> pd.DataFrame:
	return pd.DataFrame(columns=columns)


# -----------------------------
# Load & Save
# -----------------------------

def load_projects() -> pd.DataFrame:
	if os.path.exists(PROJECTS_CSV):
		df = pd.read_csv(PROJECTS_CSV, dtype=str)
		if "概算批复金额" in df.columns:
			df["概算批复金额"] = pd.to_numeric(df["概算批复金额"], errors="coerce")
		for col in ["立项时间", "（预计）开工时间", "（预计）完工时间"]:
			if col in df.columns:
				df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
		return df
	return _empty_df(PROJECT_COLUMNS)


def load_fundings() -> pd.DataFrame:
	if os.path.exists(FUNDINGS_CSV):
		df = pd.read_csv(FUNDINGS_CSV, dtype=str)
		for col in ["概算批复金额", "资金批复金额"]:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors="coerce")
		if "批复日期" in df.columns:
			df["批复日期"] = pd.to_datetime(df["批复日期"], errors="coerce").dt.date
		return df
	return _empty_df(FUNDING_COLUMNS)


def load_key_projects() -> pd.DataFrame:
	if os.path.exists(KEY_PROJECTS_CSV):
		return pd.read_csv(KEY_PROJECTS_CSV, dtype=str)
	return _empty_df(KEY_PROJECT_COLUMNS)


def load_all_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	return load_projects(), load_fundings(), load_key_projects()


def save_projects(df: pd.DataFrame) -> None:
	ensure_data_storage()
	df = df.reindex(columns=PROJECT_COLUMNS)
	df.to_csv(PROJECTS_CSV, index=False)


def save_fundings(df: pd.DataFrame) -> None:
	ensure_data_storage()
	df = df.reindex(columns=FUNDING_COLUMNS)
	df.to_csv(FUNDINGS_CSV, index=False)


def save_key_projects(df: pd.DataFrame) -> None:
	ensure_data_storage()
	df = df.reindex(columns=KEY_PROJECT_COLUMNS)
	df.to_csv(KEY_PROJECTS_CSV, index=False)


# -----------------------------
# Excel Import / Export
# -----------------------------

def export_tables_to_excel(projects: pd.DataFrame, fundings: pd.DataFrame) -> bytes:
	buffer = io.BytesIO()
	with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
		projects.to_excel(writer, index=False, sheet_name="项目基本信息")
		fundings.to_excel(writer, index=False, sheet_name="资金安排")
	buffer.seek(0)
	return buffer.read()


def export_template_excel() -> bytes:
	return export_tables_to_excel(
		pd.DataFrame(columns=PROJECT_COLUMNS),
		pd.DataFrame(columns=FUNDING_COLUMNS),
	)


def import_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
	xl = pd.ExcelFile(io.BytesIO(file_bytes))
	projects_df = _empty_df(PROJECT_COLUMNS)
	fundings_df = _empty_df(FUNDING_COLUMNS)
	for name in xl.sheet_names:
		if name.strip() == "项目基本信息":
			projects_df = xl.parse(name)
		elif name.strip() == "资金安排":
			fundings_df = xl.parse(name)
	projects_df = projects_df.reindex(columns=PROJECT_COLUMNS)
	fundings_df = fundings_df.reindex(columns=FUNDING_COLUMNS)
	if "概算批复金额" in projects_df:
		projects_df["概算批复金额"] = pd.to_numeric(projects_df["概算批复金额"], errors="coerce")
	for dcol in ["立项时间", "（预计）开工时间", "（预计）完工时间"]:
		if dcol in projects_df:
			projects_df[dcol] = pd.to_datetime(projects_df[dcol], errors="coerce").dt.date
	for c in ["概算批复金额", "资金批复金额"]:
		if c in fundings_df:
			fundings_df[c] = pd.to_numeric(fundings_df[c], errors="coerce")
	if "批复日期" in fundings_df:
		fundings_df["批复日期"] = pd.to_datetime(fundings_df["批复日期"], errors="coerce").dt.date
	if "项目状态" in projects_df:
		_projects_valid = projects_df["项目状态"].isin(PROJECT_STATUS_ALLOWED) | projects_df["项目状态"].isna()
		projects_df = projects_df[_projects_valid]
	if "资金性质" in fundings_df:
		_fund_valid = fundings_df["资金性质"].isin(FUND_NATURE_ALLOWED) | fundings_df["资金性质"].isna()
		fundings_df = fundings_df[_fund_valid]
	return projects_df, fundings_df


def validate_linkage(fundings: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
	proj_codes = set(projects["项目编码"].dropna().astype(str))
	mask = ~fundings["项目编码"].astype(str).isin(proj_codes)
	return fundings[mask]
