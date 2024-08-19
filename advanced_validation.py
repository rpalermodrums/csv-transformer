from typing import List, Dict, Any, Callable
from data_structures import CSVRow, ValidationReport, ValidationError
import re

class AdvancedValidator:
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}

    def add_rule(self, column: str, rule: Callable):
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        self.validation_rules[column].append(rule)

    def validate(self, data: List[CSVRow]) -> ValidationReport:
        report = ValidationReport()
        for row_index, row in enumerate(data):
            for column, rules in self.validation_rules.items():
                value = row[column]
                for rule in rules:
                    if not rule(value):
                        error = ValidationError(row_index, column, value, f"Failed validation rule: {rule.__name__}")
                        report.add_error(error)
        return report

# Predefined validation rules
def is_email(value: str) -> bool:
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, value) is not None

def is_phone_number(value: str) -> bool:
    phone_regex = r'^\+?1?\d{9,15}$'
    return re.match(phone_regex, value) is not None

def is_url(value: str) -> bool:
    url_regex = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
    return re.match(url_regex, value) is not None

def is_credit_card(value: str) -> bool:
    cc_regex = r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$'
    return re.match(cc_regex, value) is not None