import pandas as pd
import random
from faker import Faker


def generate_funding_excel(output_path: str = "资金安排表.xlsx", num_records: int = 1000):
    """Generate an Excel file with mock funding arrangement data.

    Args:
        output_path (str): Path to output Excel file.
        num_records (int): Number of mock records to generate.
    """

    fake = Faker("zh_CN")

    # Define fixed choices
    fund_types = [
        "一般债券",
        "专项债券",
        "中央综合财力",
        "土地出让金",
        "中央预算内投资",
        "中央补助",
        "省级补助",
    ]

    records = []
    for i in range(1, num_records + 1):
        project_name = f"{fake.city()}基础设施建设项目{i}"
        project_code = f"XM-{fake.random_uppercase_letter()}{fake.random_number(digits=5, fix_len=True)}"
        construction_unit = fake.company()
        supervising_dept = fake.job()
        estimate_approval_amount = round(random.uniform(1e6, 5e8), 2)  # 100万到5亿
        fund_approval_amount = round(
            estimate_approval_amount * random.uniform(0.8, 1.2), 2
        )
        approval_date = fake.date_between(start_date="-3y", end_date="today")
        fund_type = random.choice(fund_types)
        handler = fake.name()
        handler_department = f"{fake.word()}处室"

        record = {
            "项目名称": project_name,
            "项目编码": project_code,
            "建设单位": construction_unit,
            "项目主管部门": supervising_dept,
            "概算批复金额": estimate_approval_amount,
            "资金批复金额": fund_approval_amount,
            "批复日期": approval_date,
            "资金性质": fund_type,
            "经办人": handler,
            "经办处室": handler_department,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="资金安排表")


if __name__ == "__main__":
    generate_funding_excel()