from typing import List
from data_structures import DataType
import re
from datetime import datetime

def infer_type(values: List[str]) -> DataType:
    if all(is_integer(value) for value in values):
        return DataType.INTEGER
    if all(is_float(value) for value in values):
        return DataType.FLOAT
    if all(is_date(value) for value in values):
        return DataType.DATE
    if all(is_boolean(value) for value in values):
        return DataType.BOOLEAN
    return DataType.STRING

def is_integer(value: str) -> bool:
    return value.isdigit() or (value[0] == '-' and value[1:].isdigit())

def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_date(value: str) -> bool:
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
    for fmt in date_formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False

def is_boolean(value: str) -> bool:
    return value.lower() in ('true', 'false', 'yes', 'no', '1', '0')

def infer_column_types(data: List[CSVRow]) -> Dict[str, DataType]:
    if not data:
        return {}

    column_types = {}
    for column in data[0].keys():
        values = [row[column] for row in data]
        column_types[column] = infer_type(values)

    return column_types