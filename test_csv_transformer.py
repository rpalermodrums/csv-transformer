import unittest
from csv_reader import CSVReader, CSVParseError
from transformer import transform_csv
from data_structures import CSVRow, TransformationResult, DataType
from type_inference import infer_column_types
from advanced_processing import apply_aggregation, apply_window_function, detect_anomalies
from advanced_validation import AdvancedValidator, is_email, is_phone_number, is_url, is_credit_card
import os
import pandas as pd
import numpy as np

class TestCSVTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a sample CSV file for testing
        cls.test_csv_path = 'test_data.csv'
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Date1': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'ValueDate1': [100, 200, 300, 400, 500],
            'Currency1': ['USD', 'EUR', 'GBP', 'JPY', 'CHF'],
            'ValueCurrency1': [1.0, 1.1, 1.2, 1.3, 1.4],
            'Email': ['test1@example.com', 'test2@example.com', 'invalid_email', 'test4@example.com', 'test5@example.com'],
            'Phone': ['+1234567890', '1234567890', 'invalid_phone', '+9876543210', '5551234567'],
            'URL': ['https://example.com', 'http://test.org', 'invalid_url', 'https://github.com', 'http://linkedin.com'],
            'CreditCard': ['4111111111111111', '5500000000000004', 'invalid_card', '340000000000009', '6011000000000004'],
            'Boolean': ['True', 'False', 'Yes', 'No', '1']
        })
        df.to_csv(cls.test_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Remove the test CSV file
        os.remove(cls.test_csv_path)

    def test_csv_reader(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data[0], CSVRow)
        self.assertEqual(data[0]['ID'], '1')
        self.assertEqual(data[0]['Date1'], '2023-01-01')

    def test_csv_reader_error(self):
        with self.assertRaises(CSVParseError):
            CSVReader('non_existent_file.csv').read_all()

    def test_transform_csv(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        result = transform_csv(data, ['Date', 'Currency'])
        self.assertIsInstance(result, TransformationResult)
        self.assertEqual(len(result.data), 10)  # 5 rows * 2 transposed columns
        self.assertEqual(result.data[0]['Var'], 'Date')
        self.assertEqual(result.data[1]['Var'], 'Currency')

    def test_infer_column_types(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        column_types = infer_column_types(data)
        self.assertEqual(column_types['ID'], DataType.INTEGER)
        self.assertEqual(column_types['Date1'], DataType.DATE)
        self.assertEqual(column_types['ValueDate1'], DataType.INTEGER)
        self.assertEqual(column_types['ValueCurrency1'], DataType.FLOAT)
        self.assertEqual(column_types['Boolean'], DataType.BOOLEAN)

    def test_apply_aggregation(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        agg_result = apply_aggregation(data, ['Currency1'], {'ValueCurrency1': 'mean'})
        self.assertEqual(len(agg_result), 5)  # 5 unique currencies

    def test_apply_window_function(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        window_result = apply_window_function(data, lambda x: x.mean(), 'ValueDate1', 2)
        self.assertEqual(len(window_result), 5)
        self.assertIn('ValueDate1_window', window_result[0].keys())

    def test_detect_anomalies(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        anomaly_result = detect_anomalies(data, 'ValueDate1', method='zscore')
        self.assertEqual(len(anomaly_result), 5)
        self.assertIn('is_anomaly', anomaly_result[0].keys())

    def test_advanced_validation(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        validator = AdvancedValidator()
        validator.add_rule('Email', is_email)
        validator.add_rule('Phone', is_phone_number)
        validator.add_rule('URL', is_url)
        validator.add_rule('CreditCard', is_credit_card)
        validation_report = validator.validate(data)
        self.assertEqual(validation_report.get_error_count(), 4)  # One invalid email, phone, URL, and credit card

    def test_empty_csv(self):
        empty_csv_path = 'empty_test.csv'
        pd.DataFrame(columns=['ID', 'Value']).to_csv(empty_csv_path, index=False)
        reader = CSVReader(empty_csv_path)
        data = reader.read_all()
        self.assertEqual(len(data), 0)
        os.remove(empty_csv_path)

    def test_csv_with_missing_values(self):
        missing_values_csv_path = 'missing_values_test.csv'
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Value': [10, np.nan, 30]
        })
        df.to_csv(missing_values_csv_path, index=False)
        reader = CSVReader(missing_values_csv_path)
        data = reader.read_all()
        self.assertEqual(len(data), 3)
        self.assertEqual(data[1]['Value'], '')
        os.remove(missing_values_csv_path)

    def test_csv_with_quotes(self):
        quoted_csv_path = 'quoted_test.csv'
        with open(quoted_csv_path, 'w') as f:
            f.write('ID,Value\n1,"Hello, World!"\n2,"Quoted, Value"')
        reader = CSVReader(quoted_csv_path)
        data = reader.read_all()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['Value'], 'Hello, World!')
        self.assertEqual(data[1]['Value'], 'Quoted, Value')
        os.remove(quoted_csv_path)

    def test_transform_csv_with_invalid_column(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        result = transform_csv(data, ['InvalidColumn'])
        self.assertTrue(len(result.errors) > 0)

    def test_apply_window_function_invalid_column(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        with self.assertRaises(KeyError):
            apply_window_function(data, lambda x: x.mean(), 'InvalidColumn', 2)

    def test_detect_anomalies_invalid_method(self):
        reader = CSVReader(self.test_csv_path)
        data = reader.read_all()
        with self.assertRaises(ValueError):
            detect_anomalies(data, 'ValueDate1', method='invalid_method')

if __name__ == '__main__':
    unittest.main()