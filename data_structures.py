from typing import Dict, Any, List
from enum import Enum

class CSVRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[Any]:
        return list(self._data.values())

class DataType(Enum):
    STRING = 1
    INTEGER = 2
    FLOAT = 3
    DATE = 4
    BOOLEAN = 5

class TransformationResult:
    def __init__(self, data: List[CSVRow], errors: List[Dict[str, Any]], metadata: Dict[str, Any]):
        self.data = data
        self.errors = errors
        self.metadata = metadata

    def success_rate(self) -> float:
        total_rows = len(self.data) + len(self.errors)
        return len(self.data) / total_rows if total_rows > 0 else 0