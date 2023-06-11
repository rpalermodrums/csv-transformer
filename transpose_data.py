from typing import List
import pandas as pd
import os


def read_csv(filename: str) -> pd.DataFrame:
    """
    This function reads the CSV file into a DataFrame.

    :param filename: The name of the file to read
    :return: The DataFrame of the CSV data
    """
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        raise Exception("Error while reading CSV file: " + str(e))
    return df

def transpose_data(df: pd.DataFrame, data_types: List[str]) -> pd.DataFrame:
    """
    This function transposes the data in the DataFrame.

    :param df: The DataFrame with the original data
    :param data_types: List of data types to transpose
    :return: The DataFrame with transposed data
    """
    try:
        for data_type in data_types:
            type_columns = df.filter(like=data_type).columns
            value_columns = ["Value" + col[len(data_type) :] for col in type_columns]

            df_type_melted = df.melt(
                id_vars="ID",
                value_vars=type_columns,
                var_name="Var",
                value_name=data_type,
            )
            df_value_melted = df.melt(
                id_vars="ID",
                value_vars=value_columns,
                var_name="Var",
                value_name="Value",
            )

            df_type_melted["Var"] = df_type_melted["Var"].str.replace(data_type, "")
            df_value_melted["Var"] = df_value_melted["Var"].str.replace("Value", "")

            df_melted = pd.merge(df_type_melted, df_value_melted, on=["ID", "Var"])
            df_melted.sort_values(by=["ID", data_type], inplace=True)

            df = (
                df_melted
                if data_type == data_types[0]
                else pd.merge(df, df_melted, on=["ID", "Var", "Value"])
            )
    except KeyError as e:
        raise KeyError("Error during transposition: The DataFrame does not contain the expected columns.") from e
    except ValueError as e:
        raise ValueError("Error during transposition: The 'melt' or 'merge' operation failed.") from e
    except Exception as e:
        raise Exception("Unexpected error during transposition: " + str(e)) from e
    return df

def add_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds a unique identifier to the DataFrame.

    :param df: The DataFrame with the transposed data
    :return: The DataFrame with unique identifiers
    """
    try:
        df["UniqueID"] = range(1, len(df) + 1)
    except Exception as e:
        raise Exception("Error while adding unique identifiers: " + str(e))
    return df
