import pandas as pd
import random
from datetime import datetime, timedelta

def generate_data(num_rows: int) -> pd.DataFrame:
    """
    This function generates a DataFrame with random data for testing, including edge cases.

    :param num_rows: The number of rows to generate
    :return: The DataFrame with generated data
    """
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
    transactions = ['Withdrawal', 'Deposit', 'Transfer', 'Payment', 'Refund', 'Fee', 'Interest']
    accounts = ['Checking', 'Savings', 'Credit', 'Investment', 'Retirement', 'Loan', 'Mortgage']
    users = ['User'+str(i).zfill(3) for i in range(1, 101)]
    regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast', 'Southeast', 'Northwest', 'Southwest']
    statuses = ['Pending', 'Completed', 'Failed', 'Cancelled', 'On Hold']

    data = {"ID": range(1, num_rows + 1)}

    for i in range(1, 8):  # Increased to 7 columns per type
        # Dates
        data[f'Date{i}'] = random.choices(date_range, k=num_rows)
        data[f'ValueDate{i}'] = [random.randint(-1000, 10000) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Currencies
        data[f'Currency{i}'] = random.choices(currencies + [None], weights=[10]*len(currencies) + [1], k=num_rows)
        data[f'ValueCurrency{i}'] = [round(random.uniform(-10000, 100000), 2) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Transactions
        data[f'Transaction{i}'] = random.choices(transactions + [None], weights=[10]*len(transactions) + [1], k=num_rows)
        data[f'ValueTransaction{i}'] = [random.randint(-10000, 100000) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Accounts
        data[f'Account{i}'] = random.choices(accounts + [None], weights=[10]*len(accounts) + [1], k=num_rows)
        data[f'ValueAccount{i}'] = [random.randint(0, 1000000) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Balances
        data[f'Balance{i}'] = [round(random.uniform(-10000, 1000000), 2) if random.random() > 0.05 else None for _ in range(num_rows)]
        data[f'ValueBalance{i}'] = [round(random.uniform(-10000, 1000000), 2) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Users
        data[f'User{i}'] = random.choices(users + [None], weights=[10]*len(users) + [1], k=num_rows)
        data[f'ValueUser{i}'] = [random.randint(1, 1000) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Regions
        data[f'Region{i}'] = random.choices(regions + [None], weights=[10]*len(regions) + [1], k=num_rows)
        data[f'ValueRegion{i}'] = [random.randint(1, 100) if random.random() > 0.05 else None for _ in range(num_rows)]

        # Statuses
        data[f'Status{i}'] = random.choices(statuses + [None], weights=[10]*len(statuses) + [1], k=num_rows)
        data[f'ValueStatus{i}'] = [random.choice(['0', '1', 'True', 'False', 'Yes', 'No']) if random.random() > 0.05 else None for _ in range(num_rows)]

    # Add some invalid data
    for column in data:
        if column.startswith('Value'):
            invalid_indices = random.sample(range(num_rows), num_rows // 100)  # 1% invalid data
            for idx in invalid_indices:
                data[column][idx] = random.choice(['N/A', 'Invalid', '', 'Error', '#REF!'])

    return pd.DataFrame(data)

# Generate a DataFrame with 1000 rows of data
df = generate_data(1000)

# Save to CSV
df.to_csv("./data.csv", index=False)

print(df.head())
print(df.dtypes)