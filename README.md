# CSV Transformer Pro

CSV Transformer Pro is a powerful Python tool for processing, transforming, and analyzing CSV files. It offers a wide range of features including column transposition, data aggregation, anomaly detection, and advanced validation.

## Features

- Read and parse CSV files with support for different delimiters and quoted fields
- Transform CSV data by transposing specified columns
- Infer column data types automatically
- Apply aggregation functions to grouped data
- Perform window function operations on specified columns
- Detect anomalies using various statistical methods
- Validate data using customizable rules
- Generate advanced reports with data visualizations

## Requirements

- Docker
- Docker Compose

## Installation and Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/csv-transformer-pro.git
   cd csv-transformer-pro
   ```

2. Build the Docker image:
   ```
   docker-compose build
   ```

## Usage

Run the script using Docker Compose:

```
docker-compose up
```

You can also run the script with specific options by creating a `docker-compose.override.yml` file with the desired command:

```yaml
version: '3'
services:
  csv-transformer:
    command: python main.py input_data.csv -o output_data.csv -t Date,Value -a Currency -w ValueDate1 -d zscore -v Email,Phone
```

Then, run the script with:

```
docker-compose up
```

## Advanced Usage

For more advanced usage and customization options, refer to the inline comments in the script and the provided test suite.

## Error Handling

The script includes error handling to catch and provide informative messages for common errors that may occur during the transformation process. This includes missing columns, failed operations, and invalid input options. Any unexpected errors are also caught and raised with a general error message.

## Testing

The script includes a comprehensive test suite to ensure the correct functioning of its key components. Run the tests using the following command:

```
docker-compose exec csv-transformer python -m unittest test_csv_transformer.py
```

Ensure that all tests pass before using the script on real data. If a test fails, it indicates a potential issue with the corresponding part of the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This script is provided under the MIT License. While efforts have been made to ensure its functionality, it is provided "as is", without warranty of any kind. The authors or copyright holders shall not be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or the use or other dealings in the software. Please use at your own risk and review the [LICENSE](LICENSE) file for more details.