import csv
from typing import List, Dict, Any, Iterator
from data_structures import CSVRow

class CSVParseError(Exception):
    pass

class CSVReader:
    def __init__(self, file_path: str, delimiter: str = ',', quotechar: str = '"'):
        self.file_path = file_path
        self.delimiter = delimiter
        self.quotechar = quotechar

    def __iter__(self) -> Iterator[CSVRow]:
        try:
            with open(self.file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)
                for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header row
                    if len(row) != len(reader.fieldnames):
                        raise CSVParseError(f"Incorrect number of fields on line {row_num}")
                    yield CSVRow(row)
        except csv.Error as e:
            raise CSVParseError(f"CSV parsing error: {str(e)}")
        except IOError as e:
            raise CSVParseError(f"File I/O error: {str(e)}")

    def read_all(self) -> List[CSVRow]:
        return list(self)