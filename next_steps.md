# CSV Transformer Pro: Detailed Breakdown of Operations 1-4

## 1. Core Data Processing

### 1.1 Data Structure Design

#### 1.1.1 Define base data structure for CSV rows

We'll use a dictionary to represent each row, with column names as keys and cell values as values. This allows for flexible handling of varying column structures.

```python
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
```

#### 1.1.2 Design data structure for column metadata

Column metadata will include information such as data type, constraints, and transformation rules.

```python
from enum import Enum
from typing import Optional, Callable

class DataType(Enum):
    STRING = 1
    INTEGER = 2
    FLOAT = 3
    DATE = 4
    BOOLEAN = 5

class ColumnMetadata:
    def __init__(self, name: str, data_type: DataType, nullable: bool = True,
                 constraints: Optional[List[Callable]] = None,
                 transform: Optional[Callable] = None):
        self.name = name
        self.data_type = data_type
        self.nullable = nullable
        self.constraints = constraints or []
        self.transform = transform
```

#### 1.1.3 Create data structure for transformation results

The transformation results will include the processed data, any errors encountered, and metadata about the transformation process.

```python
class TransformationResult:
    def __init__(self, data: List[CSVRow], errors: List[Dict[str, Any]],
                 metadata: Dict[str, Any]):
        self.data = data
        self.errors = errors
        self.metadata = metadata

    def success_rate(self) -> float:
        total_rows = len(self.data) + len(self.errors)
        return len(self.data) / total_rows if total_rows > 0 else 0
```

### 1.2 CSV Parsing

#### 1.2.1 Implement basic CSV reader

We'll use Python's built-in `csv` module for basic CSV reading, but wrap it in our own class for additional functionality.

```python
import csv
from typing import List, Dict, Any, Iterator

class CSVReader:
    def __init__(self, file_path: str, delimiter: str = ',', quotechar: str = '"'):
        self.file_path = file_path
        self.delimiter = delimiter
        self.quotechar = quotechar

    def __iter__(self) -> Iterator[CSVRow]:
        with open(self.file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                yield CSVRow(row)

    def read_all(self) -> List[CSVRow]:
        return list(self)
```

#### 1.2.2 Add support for different delimiters

This is handled by the `delimiter` parameter in the `CSVReader` class above.

#### 1.2.3 Handle quoted fields

This is handled by the `quotechar` parameter in the `CSVReader` class above.

#### 1.2.4 Implement error handling for malformed CSV

We'll add error handling to catch and report issues with malformed CSV files.

```python
class CSVParseError(Exception):
    pass

class CSVReader:
    # ... (previous code remains the same)

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
```

### 1.3 Data Type Inference

#### 1.3.1 Develop algorithms for common data types

We'll implement a type inference system that examines the values in each column to determine the most likely data type.

```python
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
```

#### 1.3.2 Implement heuristics for advanced type inference

For more complex types like currency or percentages, we'll use regular expressions and additional heuristics.

```python
def infer_advanced_type(values: List[str]) -> DataType:
    if all(is_currency(value) for value in values):
        return DataType.CURRENCY
    if all(is_percentage(value) for value in values):
        return DataType.PERCENTAGE
    return infer_type(values)  # Fall back to basic type inference

def is_currency(value: str) -> bool:
    currency_pattern = r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$'
    return bool(re.match(currency_pattern, value))

def is_percentage(value: str) -> bool:
    percentage_pattern = r'^\d+(\.\d+)?%$'
    return bool(re.match(percentage_pattern, value))
```

#### 1.3.3 Handle edge cases

To handle mixed data types in a column, we'll implement a voting system that chooses the most common type, with a fallback to string if there's no clear winner.

```python
from collections import Counter

def infer_column_type(values: List[str]) -> DataType:
    type_counts = Counter(infer_advanced_type(value) for value in values)
    most_common_type, count = type_counts.most_common(1)[0]
    
    if count / len(values) >= 0.8:  # If 80% or more values are of the same type
        return most_common_type
    else:
        return DataType.STRING  # Fallback to string if mixed types
```

### 1.4 Column Transposition

#### 1.4.1 Implement basic column transposition logic

We'll implement a function to transpose specified columns from vertical to horizontal orientation.

```python
def transpose_columns(data: List[CSVRow], columns_to_transpose: List[str]) -> List[CSVRow]:
    transposed_data = []
    for row in data:
        new_row = {k: v for k, v in row._data.items() if k not in columns_to_transpose}
        for col in columns_to_transpose:
            new_row[f"{col}_{row['id']}"] = row[col]
        transposed_data.append(CSVRow(new_row))
    return transposed_data
```

#### 1.4.2 Add support for multi-column transposition

We'll extend the transposition function to handle multiple columns simultaneously.

```python
def transpose_multi_columns(data: List[CSVRow], column_groups: List[List[str]]) -> List[CSVRow]:
    transposed_data = []
    for row in data:
        new_row = {k: v for k, v in row._data.items() if not any(k in group for group in column_groups)}
        for group in column_groups:
            group_values = [row[col] for col in group]
            new_row[f"{'_'.join(group)}_{row['id']}"] = '_'.join(group_values)
        transposed_data.append(CSVRow(new_row))
    return transposed_data
```

#### 1.4.3 Develop logic for handling missing data during transposition

We'll implement a strategy to handle missing data, allowing for customizable behavior (e.g., skip, fill with default value, or raise an error).

```python
from enum import Enum

class MissingDataStrategy(Enum):
    SKIP = 1
    FILL_DEFAULT = 2
    RAISE_ERROR = 3

def transpose_columns_with_missing_data(data: List[CSVRow], columns_to_transpose: List[str],
                                        strategy: MissingDataStrategy, default_value: Any = None) -> List[CSVRow]:
    transposed_data = []
    for row in data:
        new_row = {k: v for k, v in row._data.items() if k not in columns_to_transpose}
        for col in columns_to_transpose:
            if col not in row._data or row[col] is None:
                if strategy == MissingDataStrategy.SKIP:
                    continue
                elif strategy == MissingDataStrategy.FILL_DEFAULT:
                    new_row[f"{col}_{row['id']}"] = default_value
                elif strategy == MissingDataStrategy.RAISE_ERROR:
                    raise ValueError(f"Missing data in column {col} for row {row['id']}")
            else:
                new_row[f"{col}_{row['id']}"] = row[col]
        transposed_data.append(CSVRow(new_row))
    return transposed_data
```

#### 1.4.4 Optimize transposition for large datasets

For large datasets, we'll implement a generator-based approach to reduce memory usage.

```python
from typing import Iterator

def transpose_columns_large_dataset(data: Iterator[CSVRow], columns_to_transpose: List[str]) -> Iterator[CSVRow]:
    for row in data:
        new_row = {k: v for k, v in row._data.items() if k not in columns_to_transpose}
        for col in columns_to_transpose:
            new_row[f"{col}_{row['id']}"] = row[col]
        yield CSVRow(new_row)
```

## 2. File Handling and I/O

### 2.1 Input File Handling

#### 2.1.1 Implement file reading for single CSV files

We'll create a class to handle reading single CSV files, building upon our earlier `CSVReader` class.

```python
class CSVFileHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.reader = CSVReader(file_path)

    def read_all(self) -> List[CSVRow]:
        return self.reader.read_all()

    def read_chunk(self, chunk_size: int) -> List[CSVRow]:
        chunk = []
        for row in self.reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
```

#### 2.1.2 Add support for reading multiple CSV files

We'll extend our file handling to support multiple CSV files, potentially with different structures.

```python
class MultiCSVFileHandler:
    def __init__(self, file_paths: List[str]):
        self.file_handlers = [CSVFileHandler(path) for path in file_paths]

    def read_all(self) -> Dict[str, List[CSVRow]]:
        return {handler.file_path: handler.read_all() for handler in self.file_handlers}

    def read_chunk_all(self, chunk_size: int) -> Dict[str, Iterator[List[CSVRow]]]:
        return {handler.file_path: handler.read_chunk(chunk_size) for handler in self.file_handlers}
```

#### 2.1.3 Develop streaming capabilities for large files

For large files, we'll implement a streaming approach to process data in chunks without loading the entire file into memory.

```python
class StreamingCSVHandler:
    def __init__(self, file_path: str, chunk_size: int = 1000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def stream_data(self) -> Iterator[List[CSVRow]]:
        with open(self.file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            chunk = []
            for row in reader:
                chunk.append(CSVRow(row))
                if len(chunk) == self.chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
```

#### 2.1.4 Implement error handling for file read operations

We'll add comprehensive error handling to our file reading operations.

```python
import os

class FileReadError(Exception):
    pass

class CSVFileHandler:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileReadError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise FileReadError(f"Not a file: {file_path}")
        if not file_path.lower().endswith('.csv'):
            raise FileReadError(f"Not a CSV file: {file_path}")
        
        self.file_path = file_path
        try:
            self.reader = CSVReader(file_path)
        except CSVParseError as e:
            raise FileReadError(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise FileReadError(f"Unexpected error reading file: {str(e)}")

    # ... (rest of the methods remain the same)
```


# CSV Transformer Pro: Detailed Breakdown of Operations 2-4 (Continued)

## 2. File Handling and I/O (Continued)

### 2.2 Output File Generation (Continued)

#### 2.2.1 Implement basic CSV writer (Continued)

Continuing from where we left off:

```python
import csv

class CSVWriter:
    def __init__(self, file_path: str, fieldnames: List[str]):
        self.file_path = file_path
        self.fieldnames = fieldnames

    def write_rows(self, rows: List[CSVRow]):
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row._data)

    def write_row(self, row: CSVRow):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row._data)
```

#### 2.2.2 Add support for different output formats (e.g., JSON, Parquet)

We'll create an abstract base class for writers and implement concrete classes for different formats:

```python
from abc import ABC, abstractmethod
import json
import pyarrow as pa
import pyarrow.parquet as pq

class BaseWriter(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def write_rows(self, rows: List[CSVRow]):
        pass

    @abstractmethod
    def write_row(self, row: CSVRow):
        pass

class JSONWriter(BaseWriter):
    def write_rows(self, rows: List[CSVRow]):
        with open(self.file_path, 'w') as jsonfile:
            json.dump([row._data for row in rows], jsonfile)

    def write_row(self, row: CSVRow):
        with open(self.file_path, 'a') as jsonfile:
            json.dump(row._data, jsonfile)
            jsonfile.write('\n')

class ParquetWriter(BaseWriter):
    def __init__(self, file_path: str, schema: pa.Schema):
        super().__init__(file_path)
        self.schema = schema

    def write_rows(self, rows: List[CSVRow]):
        table = pa.Table.from_pylist([row._data for row in rows], schema=self.schema)
        pq.write_table(table, self.file_path)

    def write_row(self, row: CSVRow):
        table = pa.Table.from_pylist([row._data], schema=self.schema)
        pq.write_table(table, self.file_path, append=True)
```

#### 2.2.3 Develop logic for handling large output datasets

For large datasets, we'll implement a streaming writer that writes data in chunks:

```python
class StreamingCSVWriter(BaseWriter):
    def __init__(self, file_path: str, fieldnames: List[str], chunk_size: int = 1000):
        super().__init__(file_path)
        self.fieldnames = fieldnames
        self.chunk_size = chunk_size
        self.buffer = []

    def write_rows(self, rows: List[CSVRow]):
        self.buffer.extend(rows)
        self._flush_if_needed()

    def write_row(self, row: CSVRow):
        self.buffer.append(row)
        self._flush_if_needed()

    def _flush_if_needed(self):
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if csvfile.tell() == 0:  # File is empty, write header
                writer.writeheader()
            writer.writerows([row._data for row in self.buffer])
        self.buffer = []

    def __del__(self):
        if self.buffer:
            self._flush()
```

#### 2.2.4 Implement error handling for file write operations

We'll add error handling to our write operations:

```python
class FileWriteError(Exception):
    pass

class ErrorHandlingWriter:
    def __init__(self, writer: BaseWriter):
        self.writer = writer

    def write_rows(self, rows: List[CSVRow]):
        try:
            self.writer.write_rows(rows)
        except IOError as e:
            raise FileWriteError(f"I/O error writing to file: {str(e)}")
        except Exception as e:
            raise FileWriteError(f"Unexpected error writing to file: {str(e)}")

    def write_row(self, row: CSVRow):
        try:
            self.writer.write_row(row)
        except IOError as e:
            raise FileWriteError(f"I/O error writing to file: {str(e)}")
        except Exception as e:
            raise FileWriteError(f"Unexpected error writing to file: {str(e)}")
```

### 2.3 File Compression

#### 2.3.1 Add support for reading compressed input files (e.g., gzip, zip)

We'll extend our `CSVReader` class to handle compressed files:

```python
import gzip
import zipfile
import io

class CompressedCSVReader(CSVReader):
    def __init__(self, file_path: str, compression: str = 'auto', **kwargs):
        super().__init__(file_path, **kwargs)
        self.compression = compression

    def __iter__(self) -> Iterator[CSVRow]:
        opener = self._get_opener()
        with opener(self.file_path, 'rt') as file:
            reader = csv.DictReader(file, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                yield CSVRow(row)

    def _get_opener(self):
        if self.compression == 'auto':
            if self.file_path.endswith('.gz'):
                return gzip.open
            elif self.file_path.endswith('.zip'):
                return self._open_zip
            else:
                return open
        elif self.compression == 'gzip':
            return gzip.open
        elif self.compression == 'zip':
            return self._open_zip
        else:
            return open

    def _open_zip(self, file_path, mode):
        with zipfile.ZipFile(file_path) as zf:
            return io.TextIOWrapper(zf.open(zf.namelist()[0]))
```

#### 2.3.2 Implement compression for output files

We'll create compressed writers for our output files:

```python
import gzip
import zipfile

class GzipCSVWriter(BaseWriter):
    def write_rows(self, rows: List[CSVRow]):
        with gzip.open(self.file_path, 'wt') as gzfile:
            writer = csv.DictWriter(gzfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([row._data for row in rows])

    def write_row(self, row: CSVRow):
        mode = 'at' if os.path.exists(self.file_path) else 'wt'
        with gzip.open(self.file_path, mode) as gzfile:
            writer = csv.DictWriter(gzfile, fieldnames=row.keys())
            if mode == 'wt':
                writer.writeheader()
            writer.writerow(row._data)

class ZipCSVWriter(BaseWriter):
    def __init__(self, file_path: str, internal_filename: str = 'data.csv'):
        super().__init__(file_path)
        self.internal_filename = internal_filename

    def write_rows(self, rows: List[CSVRow]):
        with zipfile.ZipFile(self.file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open(self.internal_filename, 'w') as csvfile:
                writer = csv.DictWriter(io.TextIOWrapper(csvfile, write_through=True), fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([row._data for row in rows])

    def write_row(self, row: CSVRow):
        mode = 'a' if os.path.exists(self.file_path) else 'w'
        with zipfile.ZipFile(self.file_path, mode, compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open(self.internal_filename, 'a') as csvfile:
                writer = csv.DictWriter(io.TextIOWrapper(csvfile, write_through=True), fieldnames=row.keys())
                if mode == 'w':
                    writer.writeheader()
                writer.writerow(row._data)
```

#### 2.3.3 Optimize compression/decompression process for large files

For large files, we'll implement a streaming approach for both reading and writing compressed data:

```python
class StreamingCompressedCSVReader:
    def __init__(self, file_path: str, compression: str = 'auto', chunk_size: int = 1000, **kwargs):
        self.reader = CompressedCSVReader(file_path, compression, **kwargs)
        self.chunk_size = chunk_size

    def __iter__(self):
        chunk = []
        for row in self.reader:
            chunk.append(row)
            if len(chunk) == self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

class StreamingCompressedCSVWriter:
    def __init__(self, file_path: str, compression: str = 'gzip', chunk_size: int = 1000):
        self.file_path = file_path
        self.compression = compression
        self.chunk_size = chunk_size
        self.buffer = []

    def write_rows(self, rows: List[CSVRow]):
        self.buffer.extend(rows)
        self._flush_if_needed()

    def write_row(self, row: CSVRow):
        self.buffer.append(row)
        self._flush_if_needed()

    def _flush_if_needed(self):
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self):
        opener = gzip.open if self.compression == 'gzip' else open
        mode = 'at' if os.path.exists(self.file_path) else 'wt'
        with opener(self.file_path, mode) as file:
            writer = csv.DictWriter(file, fieldnames=self.buffer[0].keys())
            if mode == 'wt':
                writer.writeheader()
            writer.writerows([row._data for row in self.buffer])
        self.buffer = []

    def __del__(self):
        if self.buffer:
            self._flush()
```

### 2.4 File Encoding Handling

#### 2.4.1 Implement automatic encoding detection

We'll use the `chardet` library to detect file encoding:

```python
import chardet

def detect_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        raw = file.read(10000)  # Read first 10000 bytes
    result = chardet.detect(raw)
    return result['encoding']

class EncodingAwareCSVReader(CSVReader):
    def __init__(self, file_path: str, encoding: str = 'auto', **kwargs):
        super().__init__(file_path, **kwargs)
        self.encoding = encoding if encoding != 'auto' else detect_encoding(file_path)

    def __iter__(self) -> Iterator[CSVRow]:
        with open(self.file_path, 'r', encoding=self.encoding, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                yield CSVRow(row)
```

#### 2.4.2 Add support for specifying custom encodings

We've already added support for custom encodings in the `EncodingAwareCSVReader` class above. Users can specify the encoding when initializing the reader.

#### 2.4.3 Handle edge cases with mixed encodings

To handle files with mixed encodings, we'll implement a more robust reading strategy:

```python
import codecs

class MixedEncodingCSVReader:
    def __init__(self, file_path: str, fallback_encodings: List[str] = ['utf-8', 'iso-8859-1', 'windows-1252']):
        self.file_path = file_path
        self.fallback_encodings = fallback_encodings

    def __iter__(self) -> Iterator[CSVRow]:
        with open(self.file_path, 'rb') as file:
            reader = self._create_reader(file)
            for row in reader:
                yield CSVRow(row)

    def _create_reader(self, file):
        raw = file.read()
        for encoding in [detect_encoding(self.file_path)] + self.fallback_encodings:
            try:
                decoded = raw.decode(encoding)
                return csv.DictReader(decoded.splitlines())
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode the file with the available encodings")

    def read_all(self) -> List[CSVRow]:
        return list(self)
```

This implementation first tries to detect the encoding automatically. If that fails, it falls back to a list of common encodings. It reads the entire file into memory, which might not be suitable for very large files, but it allows us to handle mixed encodings more effectively.

## 3. Transformation Pipeline

### 3.1 Pipeline Structure

#### 3.1.1 Design pipeline data structure

We'll create a `Pipeline` class that can hold a series of transformation steps:

```python
from typing import Callable, List

class TransformationStep:
    def __init__(self, name: str, transform_func: Callable[[List[CSVRow]], List[CSVRow]]):
        self.name = name
        self.transform_func = transform_func

class Pipeline:
    def __init__(self):
        self.steps: List[TransformationStep] = []

    def add_step(self, step: TransformationStep):
        self.steps.append(step)

    def remove_step(self, step_name: str):
        self.steps = [step for step in self.steps if step.name != step_name]

    def execute(self, data: List[CSVRow]) -> List[CSVRow]:
        for step in self.steps:
            data = step.transform_func(data)
        return data
```
#### 3.1.2 Implement mechanism to add transformation steps

The `add_step` method in the `Pipeline` class above allows adding steps. We can create a decorator to make it easier to define transformation steps:

```python
from typing import List

from generate_random_csv import generate_data

def transformation_step(name: str):
    def decorator(func):
        return TransformationStep(name, func)
    return decorator

# Usage example:
@transformation_step("Remove empty rows")
def remove_empty_rows(data: List[CSVRow]) -> List[CSVRow]:
    return [row for row in data if any(row.values())]

pipeline = Pipeline()
data = generate_data(1000)
pipeline.add_step(remove_empty_rows)
pipeline.execute(data)
```


# CSV Transformer Pro: Detailed Breakdown of Operation 3 (Continued)

## 3. Transformation Pipeline (Continued)

### 3.1 Pipeline Structure (Continued)

#### 3.1.3 Develop logic for step ordering and dependencies

To handle step ordering and dependencies, we'll extend our `Pipeline` class:

```python
from typing import Dict, Set

class DependencyError(Exception):
    pass

class Pipeline:
    def __init__(self):
        self.steps: Dict[str, TransformationStep] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def add_step(self, step: TransformationStep, dependencies: List[str] = None):
        self.steps[step.name] = step
        if dependencies:
            self.dependencies[step.name] = set(dependencies)
        else:
            self.dependencies[step.name] = set()

    def remove_step(self, step_name: str):
        if step_name in self.steps:
            del self.steps[step_name]
            del self.dependencies[step_name]
            for deps in self.dependencies.values():
                deps.discard(step_name)

    def _topological_sort(self):
        in_degree = {step: len(deps) for step, deps in self.dependencies.items()}
        queue = [step for step, degree in in_degree.items() if degree == 0]
        sorted_steps = []

        while queue:
            step = queue.pop(0)
            sorted_steps.append(step)
            for dependent in self.dependencies.keys():
                if step in self.dependencies[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(sorted_steps) != len(self.steps):
            raise DependencyError("Circular dependency detected in pipeline")

        return sorted_steps

    def execute(self, data: List[CSVRow]) -> List[CSVRow]:
        sorted_steps = self._topological_sort()
        for step_name in sorted_steps:
            data = self.steps[step_name].transform_func(data)
        return data
```

### 3.2 Step Execution

#### 3.2.1 Implement basic step execution logic

The `execute` method in our `Pipeline` class already implements basic step execution logic. Let's add error handling and logging:

```python
import logging

class StepExecutionError(Exception):
    pass

class Pipeline:
    def __init__(self):
        self.steps: Dict[str, TransformationStep] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)

    # ... (previous methods remain the same)

    def execute(self, data: List[CSVRow]) -> List[CSVRow]:
        sorted_steps = self._topological_sort()
        for step_name in sorted_steps:
            step = self.steps[step_name]
            self.logger.info(f"Executing step: {step_name}")
            try:
                data = step.transform_func(data)
            except Exception as e:
                self.logger.error(f"Error executing step {step_name}: {str(e)}")
                raise StepExecutionError(f"Error in step {step_name}: {str(e)}")
            self.logger.info(f"Completed step: {step_name}")
        return data
```

#### 3.2.2 Add support for conditional step execution

We'll modify our `TransformationStep` class to include a condition for execution:

```python
from typing import Callable, Optional

class TransformationStep:
    def __init__(self, name: str, 
                 transform_func: Callable[[List[CSVRow]], List[CSVRow]],
                 condition: Optional[Callable[[List[CSVRow]], bool]] = None):
        self.name = name
        self.transform_func = transform_func
        self.condition = condition or (lambda data: True)

class Pipeline:
    # ... (previous methods remain the same)

    def execute(self, data: List[CSVRow]) -> List[CSVRow]:
        sorted_steps = self._topological_sort()
        for step_name in sorted_steps:
            step = self.steps[step_name]
            if step.condition(data):
                self.logger.info(f"Executing step: {step_name}")
                try:
                    data = step.transform_func(data)
                except Exception as e:
                    self.logger.error(f"Error executing step {step_name}: {str(e)}")
                    raise StepExecutionError(f"Error in step {step_name}: {str(e)}")
                self.logger.info(f"Completed step: {step_name}")
            else:
                self.logger.info(f"Skipping step: {step_name} (condition not met)")
        return data
```

#### 3.2.3 Develop error handling and rollback capabilities

To implement rollback capabilities, we'll need to keep track of the state before each step:

```python
import copy

class Pipeline:
    def __init__(self):
        self.steps: Dict[str, TransformationStep] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        self.checkpoints: Dict[str, List[CSVRow]] = {}

    # ... (previous methods remain the same)

    def execute(self, data: List[CSVRow]) -> List[CSVRow]:
        sorted_steps = self._topological_sort()
        original_data = copy.deepcopy(data)
        self.checkpoints.clear()

        for step_name in sorted_steps:
            step = self.steps[step_name]
            if step.condition(data):
                self.logger.info(f"Executing step: {step_name}")
                self.checkpoints[step_name] = copy.deepcopy(data)
                try:
                    data = step.transform_func(data)
                except Exception as e:
                    self.logger.error(f"Error executing step {step_name}: {str(e)}")
                    return self.rollback(step_name, original_data)
                self.logger.info(f"Completed step: {step_name}")
            else:
                self.logger.info(f"Skipping step: {step_name} (condition not met)")
        return data

    def rollback(self, failed_step: str, original_data: List[CSVRow]) -> List[CSVRow]:
        self.logger.warning(f"Rolling back from step: {failed_step}")
        sorted_steps = self._topological_sort()
        rollback_point = sorted_steps.index(failed_step)

        if rollback_point > 0:
            previous_step = sorted_steps[rollback_point - 1]
            if previous_step in self.checkpoints:
                self.logger.info(f"Rolling back to step: {previous_step}")
                return self.checkpoints[previous_step]

        self.logger.warning("Could not find a valid rollback point. Returning original data.")
        return original_data
```

### 3.3 Pipeline Persistence

#### 3.3.1 Implement saving and loading of pipeline configurations

We'll use JSON to serialize and deserialize our pipeline configurations:

```python
import json
import importlib

class Pipeline:
    # ... (previous methods remain the same)

    def save_configuration(self, file_path: str):
        config = {
            "steps": [
                {
                    "name": step.name,
                    "module": step.transform_func.__module__,
                    "function": step.transform_func.__name__,
                    "condition_module": step.condition.__module__ if step.condition.__name__ != '<lambda>' else None,
                    "condition_function": step.condition.__name__ if step.condition.__name__ != '<lambda>' else None
                }
                for step in self.steps.values()
            ],
            "dependencies": {step: list(deps) for step, deps in self.dependencies.items()}
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_configuration(cls, file_path: str) -> 'Pipeline':
        with open(file_path, 'r') as f:
            config = json.load(f)

        pipeline = cls()
        for step_config in config['steps']:
            module = importlib.import_module(step_config['module'])
            transform_func = getattr(module, step_config['function'])
            condition = None
            if step_config['condition_module'] and step_config['condition_function']:
                condition_module = importlib.import_module(step_config['condition_module'])
                condition = getattr(condition_module, step_config['condition_function'])
            step = TransformationStep(step_config['name'], transform_func, condition)
            pipeline.add_step(step)

        pipeline.dependencies = {step: set(deps) for step, deps in config['dependencies'].items()}
        return pipeline
```

#### 3.3.2 Add versioning for pipeline configurations

We'll extend our save and load methods to include versioning:

```python
import json
import importlib
from datetime import datetime

class Pipeline:
    # ... (previous methods remain the same)

    def save_configuration(self, file_path: str):
        config = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "steps": [
                {
                    "name": step.name,
                    "module": step.transform_func.__module__,
                    "function": step.transform_func.__name__,
                    "condition_module": step.condition.__module__ if step.condition.__name__ != '<lambda>' else None,
                    "condition_function": step.condition.__name__ if step.condition.__name__ != '<lambda>' else None
                }
                for step in self.steps.values()
            ],
            "dependencies": {step: list(deps) for step, deps in self.dependencies.items()}
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_configuration(cls, file_path: str) -> 'Pipeline':
        with open(file_path, 'r') as f:
            config = json.load(f)

        if config.get("version") != "1.0":
            raise ValueError(f"Unsupported configuration version: {config.get('version')}")

        pipeline = cls()
        for step_config in config['steps']:
            module = importlib.import_module(step_config['module'])
            transform_func = getattr(module, step_config['function'])
            condition = None
            if step_config['condition_module'] and step_config['condition_function']:
                condition_module = importlib.import_module(step_config['condition_module'])
                condition = getattr(condition_module, step_config['condition_function'])
            step = TransformationStep(step_config['name'], transform_func, condition)
            pipeline.add_step(step)

        pipeline.dependencies = {step: set(deps) for step, deps in config['dependencies'].items()}
        return pipeline
```

#### 3.3.3 Develop import/export functionality for sharing pipelines

We'll create methods to export pipelines as standalone Python scripts and import them:

```python
import inspect

class Pipeline:
    # ... (previous methods remain the same)

    def export_as_script(self, file_path: str):
        with open(file_path, 'w') as f:
            f.write("from csv_transformer_pro import Pipeline, TransformationStep\n\n")
            f.write("def create_pipeline():\n")
            f.write("    pipeline = Pipeline()\n\n")
            for step_name, step in self.steps.items():
                f.write(f"    def {step_name}_func(data):\n")
                f.write(inspect.getsource(step.transform_func))
                f.write("\n")
                if step.condition.__name__ != '<lambda>':
                    f.write(f"    def {step_name}_condition(data):\n")
                    f.write(inspect.getsource(step.condition))
                    f.write("\n")
                    f.write(f"    pipeline.add_step(TransformationStep('{step_name}', {step_name}_func, {step_name}_condition))\n")
                else:
                    f.write(f"    pipeline.add_step(TransformationStep('{step_name}', {step_name}_func))\n")
            f.write("\n    return pipeline\n")

    @classmethod
    def import_from_script(cls, file_path: str) -> 'Pipeline':
        with open(file_path, 'r') as f:
            script_contents = f.read()
        
        local_vars = {}
        exec(script_contents, globals(), local_vars)
        create_pipeline_func = local_vars.get('create_pipeline')
        
        if not create_pipeline_func:
            raise ValueError("The script does not contain a create_pipeline function")
        
        return create_pipeline_func()
```

### 3.4 Custom Step Creation

#### 3.4.1 Design interface for custom step creation

We'll create a base class for custom steps:

```python
from abc import ABC, abstractmethod

class CustomStep(ABC):
    @abstractmethod
    def transform(self, data: List[CSVRow]) -> List[CSVRow]:
        pass

    @abstractmethod
    def condition(self, data: List[CSVRow]) -> bool:
        pass

class Pipeline:
    # ... (previous methods remain the same)

    def add_custom_step(self, custom_step: CustomStep):
        step = TransformationStep(
            custom_step.__class__.__name__,
            custom_step.transform,
            custom_step.condition
        )
        self.add_step(step)
```

#### 3.4.2 Implement validation for custom steps

We'll add a validation method to ensure custom steps meet our requirements:

```python
import inspect

class Pipeline:
    # ... (previous methods remain the same)

    def add_custom_step(self, custom_step: CustomStep):
        self._validate_custom_step(custom_step)
        step = TransformationStep(
            custom_step.__class__.__name__,
            custom_step.transform,
            custom_step.condition
        )
        self.add_step(step)

    def _validate_custom_step(self, custom_step: CustomStep):
        if not isinstance(custom_step, CustomStep):
            raise TypeError("Custom step must inherit from CustomStep class")

        transform_sig = inspect.signature(custom_step.transform)
        condition_sig = inspect.signature(custom_step.condition)

        if len(transform_sig.parameters) != 1:
            raise ValueError("transform method must have exactly one parameter")

        if len(condition_sig.parameters) != 1:
            raise ValueError("condition method must have exactly one parameter")

        if transform_sig.return_annotation != List[CSVRow]:
            raise TypeError("transform method must return List[CSVRow]")

        if condition_sig.return_annotation != bool:
            raise TypeError("condition method must return bool")
```

# CSV Transformer Pro: Detailed Breakdown of Operations 3.4 and 4

## 3. Transformation Pipeline (Continued)

### 3.4 Custom Step Creation (Continued)

#### 3.4.3 Develop documentation generation for custom steps (Continued)

Continuing from where we left off:

```python
import inspect

class Pipeline:
    # ... (previous methods remain the same)

    def generate_custom_step_documentation(self, custom_step: CustomStep) -> str:
        doc = f"# Custom Step: {custom_step.__class__.__name__}\n\n"
        
        class_doc = inspect.getdoc(custom_step.__class__)
        if class_doc:
            doc += f"{class_doc}\n\n"
        
        doc += "## Transform Method\n\n"
        transform_doc = inspect.getdoc(custom_step.transform)
        if transform_doc:
            doc += f"{transform_doc}\n\n"
        
        doc += "### Signature\n\n"
        doc += f"```python\n{inspect.getsource(custom_step.transform)}```\n\n"
        
        doc += "## Condition Method\n\n"
        condition_doc = inspect.getdoc(custom_step.condition)
        if condition_doc:
            doc += f"{condition_doc}\n\n"
        
        doc += "### Signature\n\n"
        doc += f"```python\n{inspect.getsource(custom_step.condition)}```\n\n"
        
        return doc

    def generate_pipeline_documentation(self) -> str:
        doc = "# Pipeline Documentation\n\n"
        
        for step_name, step in self.steps.items():
            doc += f"## Step: {step_name}\n\n"
            
            if isinstance(step.transform_func, CustomStep):
                doc += self.generate_custom_step_documentation(step.transform_func)
            else:
                doc += f"### Transform Function\n\n```python\n{inspect.getsource(step.transform_func)}```\n\n"
                
                if step.condition.__name__ != '<lambda>':
                    doc += f"### Condition Function\n\n```python\n{inspect.getsource(step.condition)}```\n\n"
                else:
                    doc += "### Condition Function\n\nDefault condition (always true)\n\n"
            
            doc += f"### Dependencies\n\n"
            if step_name in self.dependencies:
                doc += ", ".join(self.dependencies[step_name]) + "\n\n"
            else:
                doc += "No dependencies\n\n"
        
        return doc
```

This method generates comprehensive documentation for each step in the pipeline, including custom steps. It extracts docstrings, function signatures, and source code to provide a clear understanding of each step's functionality.

## 4. Data Validation and Cleansing

### 4.1 Data Validation Rules

#### 4.1.1 Implement basic validation rules (e.g., data type, range, regex)

We'll create a `ValidationRule` class to represent different types of validation rules:

```python
from abc import ABC, abstractmethod
import re
from datetime import datetime

class ValidationRule(ABC):
    @abstractmethod
    def validate(self, value: Any) -> bool:
        pass

    @abstractmethod
    def get_error_message(self, value: Any) -> str:
        pass

class DataTypeRule(ValidationRule):
    def __init__(self, data_type: Type):
        self.data_type = data_type

    def validate(self, value: Any) -> bool:
        return isinstance(value, self.data_type)

    def get_error_message(self, value: Any) -> str:
        return f"Expected type {self.data_type.__name__}, got {type(value).__name__}"

class RangeRule(ValidationRule):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> bool:
        try:
            return self.min_value <= float(value) <= self.max_value
        except ValueError:
            return False

    def get_error_message(self, value: Any) -> str:
        return f"Value {value} is not within the range [{self.min_value}, {self.max_value}]"

class RegexRule(ValidationRule):
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def validate(self, value: Any) -> bool:
        return bool(self.pattern.match(str(value)))

    def get_error_message(self, value: Any) -> str:
        return f"Value {value} does not match the pattern {self.pattern.pattern}"

class DateFormatRule(ValidationRule):
    def __init__(self, date_format: str):
        self.date_format = date_format

    def validate(self, value: Any) -> bool:
        try:
            datetime.strptime(str(value), self.date_format)
            return True
        except ValueError:
            return False

    def get_error_message(self, value: Any) -> str:
        return f"Value {value} does not match the date format {self.date_format}"
```

#### 4.1.2 Add support for custom validation rules

We can allow users to create custom validation rules by subclassing `ValidationRule`:

```python
class CustomValidationRule(ValidationRule):
    def __init__(self, validate_func: Callable[[Any], bool], error_message: str):
        self.validate_func = validate_func
        self.error_message = error_message

    def validate(self, value: Any) -> bool:
        return self.validate_func(value)

    def get_error_message(self, value: Any) -> str:
        return self.error_message.format(value=value)

# Example usage:
even_number_rule = CustomValidationRule(
    lambda x: int(x) % 2 == 0,
    "Value {value} is not an even number"
)
```

#### 4.1.3 Develop composite validation rules

We'll create a `CompositeValidationRule` class that can combine multiple rules:

```python
class CompositeValidationRule(ValidationRule):
    def __init__(self, rules: List[ValidationRule], operator: str = 'AND'):
        self.rules = rules
        self.operator = operator.upper()
        if self.operator not in ['AND', 'OR']:
            raise ValueError("Operator must be either 'AND' or 'OR'")

    def validate(self, value: Any) -> bool:
        if self.operator == 'AND':
            return all(rule.validate(value) for rule in self.rules)
        else:  # OR
            return any(rule.validate(value) for rule in self.rules)

    def get_error_message(self, value: Any) -> str:
        failed_rules = [rule for rule in self.rules if not rule.validate(value)]
        error_messages = [rule.get_error_message(value) for rule in failed_rules]
        return f"Failed {self.operator} composite validation: {'; '.join(error_messages)}"
```

### 4.2 Data Cleansing Operations

#### 4.2.1 Implement basic cleansing operations (e.g., trimming, case normalization)

We'll create a `CleansingOperation` class to represent different types of cleansing operations:

```python
from abc import ABC, abstractmethod

class CleansingOperation(ABC):
    @abstractmethod
    def clean(self, value: Any) -> Any:
        pass

class TrimOperation(CleansingOperation):
    def clean(self, value: Any) -> str:
        return str(value).strip()

class LowerCaseOperation(CleansingOperation):
    def clean(self, value: Any) -> str:
        return str(value).lower()

class UpperCaseOperation(CleansingOperation):
    def clean(self, value: Any) -> str:
        return str(value).upper()

class ReplaceOperation(CleansingOperation):
    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new

    def clean(self, value: Any) -> str:
        return str(value).replace(self.old, self.new)
```

#### 4.2.2 Add advanced cleansing operations (e.g., outlier detection, imputation)

We'll implement more sophisticated cleansing operations:

```python
import numpy as np
from scipy import stats

class OutlierRemovalOperation(CleansingOperation):
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold

    def clean(self, values: List[float]) -> List[float]:
        z_scores = np.abs(stats.zscore(values))
        return [v for v, z in zip(values, z_scores) if z <= self.z_threshold]

class ImputationOperation(CleansingOperation):
    def __init__(self, strategy: str = 'mean'):
        self.strategy = strategy

    def clean(self, values: List[float]) -> List[float]:
        non_null_values = [v for v in values if v is not None]
        if self.strategy == 'mean':
            impute_value = np.mean(non_null_values)
        elif self.strategy == 'median':
            impute_value = np.median(non_null_values)
        elif self.strategy == 'mode':
            impute_value = stats.mode(non_null_values).mode[0]
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}")
        
        return [v if v is not None else impute_value for v in values]
```

#### 4.2.3 Develop custom cleansing function support

We'll allow users to define custom cleansing operations:

```python
class CustomCleansingOperation(CleansingOperation):
    def __init__(self, clean_func: Callable[[Any], Any]):
        self.clean_func = clean_func

    def clean(self, value: Any) -> Any:
        return self.clean_func(value)

# Example usage:
remove_special_chars = CustomCleansingOperation(
    lambda x: ''.join(c for c in str(x) if c.isalnum() or c.isspace())
)
```

### 4.3 Validation Reporting

#### 4.3.1 Implement basic validation error reporting

We'll create a `ValidationReport` class to store and display validation results:

```python
class ValidationError:
    def __init__(self, row_index: int, column_name: str, value: Any, error_message: str):
        self.row_index = row_index
        self.column_name = column_name
        self.value = value
        self.error_message = error_message

class ValidationReport:
    def __init__(self):
        self.errors: List[ValidationError] = []

    def add_error(self, error: ValidationError):
        self.errors.append(error)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_error_count(self) -> int:
        return len(self.errors)

    def get_errors_by_column(self) -> Dict[str, List[ValidationError]]:
        errors_by_column = defaultdict(list)
        for error in self.errors:
            errors_by_column[error.column_name].append(error)
        return dict(errors_by_column)

    def __str__(self) -> str:
        if not self.has_errors():
            return "No validation errors found."
        
        report = f"Validation Report: {self.get_error_count()} errors found\n\n"
        for column, errors in self.get_errors_by_column().items():
            report += f"Column: {column}\n"
            for error in errors:
                report += f"  Row {error.row_index}: {error.error_message} (value: {error.value})\n"
            report += "\n"
        return report
```

#### 4.3.2 Add detailed validation statistics

We'll extend the `ValidationReport` class to include more detailed statistics:

```python
from collections import Counter

class ValidationReport:
    # ... (previous methods remain the same)

    def get_error_distribution(self) -> Dict[str, int]:
        return Counter(error.column_name for error in self.errors)

    def get_error_rate(self, total_rows: int) -> float:
        return len(set(error.row_index for error in self.errors)) / total_rows

    def get_column_error_rates(self, total_rows: int) -> Dict[str, float]:
        error_distribution = self.get_error_distribution()
        return {column: count / total_rows for column, count in error_distribution.items()}

    def get_summary(self, total_rows: int) -> str:
        summary = f"Validation Summary:\n"
        summary += f"Total rows: {total_rows}\n"
        summary += f"Total errors: {self.get_error_count()}\n"
        summary += f"Overall error rate: {self.get_error_rate(total_rows):.2%}\n\n"
        summary += "Error distribution by column:\n"
        for column, rate in self.get_column_error_rates(total_rows).items():
            summary += f"  {column}: {rate:.2%}\n"
        return summary
```

#### 4.3.3 Develop visual representation of validation results

We'll create a method to generate a simple ASCII-based visualization of the validation results:

```python
class ValidationReport:
    # ... (previous methods remain the same)

    def generate_ascii_visualization(self, total_rows: int, width: int = 50) -> str:
        visualization = "Validation Results Visualization:\n\n"
        error_rates = self.get_column_error_rates(total_rows)
        
        for column, rate in error_rates.items():
            bar_length = int(rate * width)
            bar = f"{'#' * bar_length}{'-' * (width - bar_length)}"
            visualization += f"{column[:20]:20} [{bar}] {rate:.2%}\n"
        
        return visualization
```

This ASCII visualization provides a quick overview of the error rates for each column, making it easy to identify problematic areas in the data.
