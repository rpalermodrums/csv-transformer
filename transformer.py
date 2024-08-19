from typing import List, Dict
from data_structures import CSVRow, TransformationResult
from type_inference import infer_column_types

def transpose_columns(data: List[CSVRow], columns_to_transpose: List[str]) -> List[CSVRow]:
    if not data:
        return []

    transposed_data = []
    for row in data:
        for column in columns_to_transpose:
            new_row = CSVRow({
                'ID': row['ID'],
                'Var': column,
                column: row[column],
                'Value': row[f'Value{column}'],
            })
            transposed_data.append(new_row)

    return transposed_data

def transform_csv(data: List[CSVRow], columns_to_transpose: List[str]) -> TransformationResult:
    try:
        column_types = infer_column_types(data)
        transposed_data = transpose_columns(data, columns_to_transpose)
        
        metadata = {
            'original_columns': list(data[0].keys()),
            'transposed_columns': columns_to_transpose,
            'column_types': {col: str(dtype) for col, dtype in column_types.items()}
        }
        
        return TransformationResult(transposed_data, [], metadata)
    except Exception as e:
        return TransformationResult([], [{'error': str(e)}], {})