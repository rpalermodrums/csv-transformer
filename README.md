# Transpose Columns from CSV Files
This Python script reads CSV files, transposes specific columns from vertical to horizontal, matches information from the columns to the transposed data, adds unique identifiers for each new row, and produces CSV outputs.
## Requirements
Python 3.6 or higher
pandas library
## Installation
Install the required libraries using pip:
```
pip install pandas
```
## Usage
Place the CSV file in the same directory as the Python script. Run the script from the command line:
```
python transform_csv.py <filename.csv>
```
This will create an output file called "output.csv" in the same directory.

## Input Format
The script expects the input CSV files to have a specific format:
* The first column should be an ID column.
* There should be multiple pairs of columns for each data type. For example, Date1 and Value1, Date2 and Value2, etc. The Date columns contain dates and the Value columns contain corresponding values.
## Output Format
The output CSV file will have at least the following columns:
* ID: The ID from the input data.
* Var: The original column number from the input data.
* Date: The date from the corresponding Date column in the input data.
* Value: The value from the corresponding Value column in the input data.
* UniqueID: A unique identifier for each row.
## Customization
You may need to customize the script based on the actual format of your CSV files. For example, you may need to change the way it identifies date columns, or the way it transposes and matches data. This can be done by adjusting the data_types list in the transpose_data function.
## Error Handling
The script includes error handling to catch and report errors during the execution of each function. If an error occurs, the script will stop and print an error message. If you encounter an error, check the error message and the accompanying traceback for clues about what went wrong. Also, ensure that your input CSV file is in the correct format as specified above.
## Testing
The script includes unit tests to ensure the correct functioning of its key components. Run the tests using the following command:
```
python -m unittest test_transpose_data.py
```
Ensure that all tests pass before using the script on real data. If a test fails, it indicates a potential issue with the corresponding part of the script.
## Disclaimer
*This script is provided as is, without warranties of any kind. The author is not responsible for any loss or damage resulting from the use of this script. Please use at your own risk.*
