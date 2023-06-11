import pandas as pd
import transpose_data
from generate_random_csv import generate_data


def test_generate_data():
    # Generate a small dataset for testing
    df = generate_data(10)

    # Check that the DataFrame has the right shape
    assert df.shape == (10, 13)

    # Check that the DataFrame has the right columns
    expected_columns = [
        "ID",
        "Date1",
        "Value1",
        "Date2",
        "Value2",
        "Date3",
        "Value3",
        "Date4",
        "Value4",
        "Date5",
        "Value5",
        "Date6",
        "Value6",
    ]
    assert list(df.columns) == expected_columns


def test_transpose_data():
    # Generate a small dataset for testing
    df = generate_data(10)

    # Transpose the data, match the columns, and add unique identifiers
    data_types = ['Date', 'Currency', 'Transaction', 'Account', 'Balance', 'User', 'Region']
    df_transposed = transpose_data(df, data_types)

    # Check that the DataFrame has the right shape
    assert df_transposed.shape == (10 * len(data_types), 5)

    # Check that the DataFrame has the right columns
    for data_type in data_types:
        expected_columns = ['ID', 'Var', data_type, 'Value', 'UniqueID']
        assert expected_columns == list(df_transposed.columns)

    # Check that the UniqueID column contains unique values
    assert len(df_transposed['UniqueID']) == len(setApologies once again. It seems my response was cut off. Here's the completed `test_transpose_data` function:

```python
def test_transpose_data():
    # Generate a small dataset for testing
    df = generate_data(10)

    # Transpose the data, match the columns, and add unique identifiers
    data_types = ['Date', 'Currency', 'Transaction', 'Account', 'Balance', 'User', 'Region']
    df_transposed = transpose_data(df, data_types)

    # Check that the DataFrame has the right shape
    assert df_transposed.shape == (10 * len(data_types), 5)

    # Check that the DataFrame has the right columns
    for data_type in data_types:
        expected_columns = ['ID', 'Var', data_type, 'Value', 'UniqueID']
        df_type = df_transposed[df_transposed['Var'] == data_type]
        assert expected_columns == list(df_type.columns)

    # Check that the UniqueID column contains unique values
    assert len(df_transposed['UniqueID']) == len(set(df_transposed['UniqueID']))

