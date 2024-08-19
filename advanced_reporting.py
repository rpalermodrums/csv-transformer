from typing import List, Dict, Any
from data_structures import CSVRow, ValidationReport
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AdvancedReporter:
    def __init__(self, data: List[CSVRow], validation_report: ValidationReport):
        self.data = data
        self.validation_report = validation_report
        self.df = pd.DataFrame([row._data for row in data])

    def generate_summary_statistics(self) -> Dict[str, Dict[str, Any]]:
        return self.df.describe().to_dict()

    def generate_correlation_matrix(self, columns: List[str]) -> pd.DataFrame:
        return self.df[columns].corr()

    def plot_histogram(self, column: str, output_file: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, kde=True)
        plt.title(f'Histogram of {column}')
        plt.savefig(output_file)
        plt.close()

    def plot_scatter(self, x_column: str, y_column: str, output_file: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=x_column, y=y_column)
        plt.title(f'Scatter plot of {x_column} vs {y_column}')
        plt.savefig(output_file)
        plt.close()

    def plot_validation_results(self, output_file: str):
        error_distribution = self.validation_report.get_error_distribution()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(error_distribution.keys()), y=list(error_distribution.values()))
        plt.title('Validation Errors by Column')
        plt.xlabel('Column')
        plt.ylabel('Number of Errors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def generate_html_report(self, output_file: str):
        html_content = f"""
        <html>
        <head>
            <title>CSV Transformer Pro - Advanced Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>CSV Transformer Pro - Advanced Report</h1>
            
            <h2>Summary Statistics</h2>
            {pd.DataFrame(self.generate_summary_statistics()).to_html()}
            
            <h2>Validation Results</h2>
            <p>Total Errors: {self.validation_report.get_error_count()}</p>
            <img src="validation_results.png" alt="Validation Results">
            
            <h2>Data Visualizations</h2>
            <img src="histogram.png" alt="Histogram">
            <img src="scatter_plot.png" alt="Scatter Plot">
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)