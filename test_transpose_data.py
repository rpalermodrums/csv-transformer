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

    # List of data types to transpose
    data_types = [
        "Date",
        "Type1",
        "Type2",
        "Type3",
        "Type4",
        "Type5",
        "Type6",
        "Type7",
        "Type8",
        "Type9",
    ]

    # Transpose the data and add unique identifiers
    df_transposed = transpose_data(df, data_types)
    df_with_uids = add_unique_id(df_transposed)

    # Check that the DataFrame has the right shape
    assert df_with_uids.shape == (600, 5)

    # Check that the DataFrame has the right columns
    expected_columns = ["ID", "Var", "Date", "Value", "UniqueID"]
    assert list(df_with_uids.columns) == expected_columns

    # Check that the UniqueID column contains unique values
    assert len(df_with_uids["UniqueID"]) == len(set(df_with_uids["UniqueID"]))
