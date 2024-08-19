import argparse
from csv_reader import CSVReader, CSVParseError
from transformer import transform_csv
from advanced_processing import apply_aggregation, apply_window_function, detect_anomalies
from advanced_validation import AdvancedValidator, is_email, is_phone_number
from advanced_reporting import AdvancedReporter

def main():
    parser = argparse.ArgumentParser(description="CSV Transformer Pro")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--transpose", nargs='+', help="Columns to transpose")
    parser.add_argument("--aggregate", nargs='+', help="Columns to group by for aggregation")
    parser.add_argument("--window", nargs=3, metavar=('COLUMN', 'FUNCTION', 'SIZE'), help="Apply window function")
    parser.add_argument("--detect-anomalies", nargs=2, metavar=('COLUMN', 'METHOD'), help="Detect anomalies")
    parser.add_argument("--validate", action="store_true", help="Perform advanced validation")
    parser.add_argument("--report", action="store_true", help="Generate advanced report")
    args = parser.parse_args()

    try:
        reader = CSVReader(args.input_file)
        data = reader.read_all()
        
        if args.transpose:
            result = transform_csv(data, args.transpose)
            data = result.data
        
        if args.aggregate:
            agg_func = {'Value': 'sum'}  # Example aggregation function
            data = apply_aggregation(data, args.aggregate, agg_func)
        
        if args.window:
            column, func, size = args.window
            window_func = {'mean': lambda x: x.mean(), 'sum': lambda x: x.sum()}[func]
            data = apply_window_function(data, window_func, column, int(size))
        
        if args.detect_anomalies:
            column, method = args.detect_anomalies
            data = detect_anomalies(data, column, method)
        
        if args.validate:
            validator = AdvancedValidator()
            validator.add_rule('Email', is_email)
            validator.add_rule('Phone', is_phone_number)
            validation_report = validator.validate(data)
            print(validation_report)
        
        if args.report:
            reporter = AdvancedReporter(data, validation_report)
            reporter.plot_validation_results('validation_results.png')
            reporter.plot_histogram('Value', 'histogram.png')
            reporter.plot_scatter('ID', 'Value', 'scatter_plot.png')
            reporter.generate_html_report('advanced_report.html')
            print("Advanced report generated: advanced_report.html")
        
        # Save the processed data
        output_file = "output_advanced.csv"
        with open(output_file, 'w') as f:
            f.write(','.join(data[0].keys()) + '\n')
            for row in data:
                f.write(','.join(str(v) for v in row.values()) + '\n')
        print(f"Processed data saved to {output_file}")

    except CSVParseError as e:
        print(f"Error parsing CSV: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()