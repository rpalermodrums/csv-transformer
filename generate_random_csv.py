from typing import List
import pandas as pd
import random


def generate_data(num_rows):
    date_range = pd.date_range(start='1/1/2020', end='1/01/2023')
    data_types = ['Date', 'Currency', 'Transaction', 'Account', 'Balance', 'User', 'Region']
    data = {
        "ID": range(1, num_rows + 1),
    }
    for data_type in data_types:
        for i in range(Apologies, it seems that my response was cut off. Here's the completed `generate_data` function:

```python
def generate_data(num_rows: int) -> pd.DataFrame:
    """
    This function generates a DataFrame with random data for testing.

    :param num_rows: The number of rows to generate
    :return: The DataFrame with generated data
    """
    date_range = pd.date_range(start='1/1/2020', end='1/01/2023')
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    transactions = ['Withdrawal', 'Deposit', 'Transfer', 'Payment']
    accounts = ['Checking', 'Savings', 'Credit', 'Investment']
    users = ['User'+str(i) for i in range(1, 21)]
    regions = ['North', 'South', 'East', 'West']

    data = {"ID": range(1, num_rows + 1)}

    for i in range(1, 7):
        data[f'Date{i}'] = random.choices(date_range, k=num_rows)
        data[f'ValueDate{i}'] = random.choices(range(100, 1000), k=num_rows)
        data[f'Currency{i}'] = random.choices(currencies, k=num_rows)
        data[f'ValueCurrency{i}'] = random.choices(range(100, 1000), k=num_rows)
        data[f'Transaction{i}'] = random.choices(transactions, k=num_rows)
        data[f'ValueTransaction{i}'] = random.choices(range(100, 1000), k=num_rows)
        data[f'Account{i}'] = random.choices(accounts, k=num_rows)
        data[f'ValueAccount{i}'] = random.choices(range(100, 1000), k=num_rows)
        data[f'Balance{i}'] = random.choices(range(1000, 10000), k=num_rows)
        data[f'ValueBalance{i}'] = random.choices(range(1000, 10000), k=num_rows)
        data[f'User{i}'] = random.choices(users, k=num_rows)
        data[f'ValueUser{i}'] = random.choices(range(100, 1000), k=num_rows)
        data[f'Region{i}'] = random.choices(regions, k=num_rows)
        data[f'ValueRegion{i}'] = random.choices(range(100, 1000), k=num_rows)

    return pd.DataFrame(data)



# Generate a DataFrame with 1000 rows of data
df = generate_data(1000)

# Save to CSV
df.to_csv("./data.csv", index=False)

df
