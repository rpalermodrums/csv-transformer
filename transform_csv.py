from typing import Union
from transpose_data import transpose_data

# TODO: This is for testing purposes only, there should not be a default filename in production for safety reasons
dangerous_testing_filename_do_not_use_in_prod = "./data.csv"


def transform_csv(filename: str, data_types: Union[str, None] = []) -> str:
    """
    This function retrieves transposed data and creates a relevant csv from the DataFrame. It's intended to be called from the command line, as the main runner for the program.

    :param filename: The DataFrame with the original data
    :param data_types: List of data types to transpose in a given DataFrame
    :return: The DataFrame with transposed data as a csv file, stored in the current directory
    """
    df_melted = transpose_data(filename)
    df_melted.to_csv("./output.csv", index=False)

    if __name__ == "__main__":
        df_melted_csv = transform_csv(
            filename=dangerous_testing_filename_do_not_use_in_prod
        )
        return df_melted_csv
