import random
from datetime import datetime, timedelta, date

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from faker import Faker


def random_date(start_date: date, end_date: date) -> date:
    delta_days = (end_date - start_date).days
    rand_days = random.randint(0, delta_days)
    return start_date + timedelta(days=rand_days)


def generate_project_code() -> str:
    prefix = "PRJ"
    date_part = datetime.now().strftime("%Y%m%d")
    serial = f"{random.randint(0, 9999):04d}"
    return f"{prefix}{date_part}{serial}"


def build_workbook(num_rows: int = 1000) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "资金安排表"

    headers = [
        "项目名称",
        "项目编码",
        "建设单位",
        "项目主管部门",
        "概算批复金额",
        "资金批复金额",
        "批复日期",
        "资金性质",
        "经办人",
        "经办处室",
    ]
    ws.append(headers)

    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="FFDDEBF7", end_color="FFDDEBF7", fill_type="solid")
    thin = Side(border_style="thin", color="FFAAAAAA")
    header_border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_alignment = Alignment(horizontal="center", vertical="center")

    for col_idx, _ in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = header_border
        cell.alignment = header_alignment

    widths = [24, 22, 22, 22, 16, 16, 14, 18, 12, 18]
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + i)].width = width

    ws.freeze_panes = "A2"

    allowed_fund_types = ["一般债券", "专项债券", "中央综合财力", "土地出让金", "中央预算内投资", "中央补助", "省级补助"]
    validation_list = '"' + ",".join(allowed_fund_types) + '"'
    dv = DataValidation(
        type="list",
        formula1=validation_list,
        allow_blank=True,
        showErrorMessage=True,
        error="请选择预设的资金性质",
        errorTitle="输入无效",
    )
    ws.add_data_validation(dv)
    dv.add(f"H2:H{num_rows + 1}")

    amount_number_format = "#,##0.00"
    date_number_format = "yyyy-mm-dd"

    faker = Faker("zh_CN")

    start_date = date(2018, 1, 1)
    end_date = date(2025, 12, 31)

    departments = [
        "发展改革局",
        "财政局",
        "住建局",
        "交通运输局",
        "农业农村局",
        "教育局",
        "卫生健康委",
        "生态环境局",
        "水利局",
    ]

    offices = [
        "投资管理处",
        "综合处",
        "预算处",
        "项目管理处",
        "资金监管处",
        "规划处",
        "建设管理处",
    ]

    for _ in range(num_rows):
        project_name = f"{faker.city()}-{faker.word()}建设项目"
        project_code = generate_project_code()
        construction_unit = faker.company()
        supervising_dept = random.choice(departments)
        estimate_amount = round(random.uniform(5_000_000, 800_000_000), 2)
        approved_amount = max(0.0, round(estimate_amount * random.uniform(0.8, 1.1), 2))
        approval_date = random_date(start_date, end_date)
        fund_type = random.choice(allowed_fund_types)
        handler = faker.name()
        office = random.choice(offices)

        row = [
            project_name,
            project_code,
            construction_unit,
            supervising_dept,
            estimate_amount,
            approved_amount,
            approval_date,
            fund_type,
            handler,
            office,
        ]
        ws.append(row)

    for r in range(2, num_rows + 2):
        ws.cell(row=r, column=5).number_format = amount_number_format
        ws.cell(row=r, column=6).number_format = amount_number_format
        ws.cell(row=r, column=7).number_format = date_number_format

    ws.auto_filter.ref = f"A1:J{num_rows + 1}"

    return wb


def main():
    wb = build_workbook(1000)
    output_path = "/workspace/资金安排表.xlsx"
    wb.save(output_path)
    print(f"文件已生成: {output_path}")


if __name__ == "__main__":
    main()

