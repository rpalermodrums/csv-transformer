from typing import List
import pandas as pd
import random


def generate_data(num_rows):
    date_range = pd.date_range(start="1/1/2020", end="1/01/2023")
    data = {
        "ID": range(1, num_rows + 1),
        "Date1": random.choices(date_range, k=num_rows),
        "Value1": random.choices(range(100, 1000), k=num_rows),
        "Date2": random.choices(date_range, k=num_rows),
        "Value2": random.choices(range(100, 1000), k=num_rows),
        "Date3": random.choices(date_range, k=num_rows),
        "Value3": random.choices(range(100, 1000), k=num_rows),
        "Date4": random.choices(date_range, k=num_rows),
        "Value4": random.choices(range(100, 1000), k=num_rows),
        "Date5": random.choices(date_range, k=num_rows),
        "Value5": random.choices(range(100, 1000), k=num_rows),
        "Date6": random.choices(date_range, k=num_rows),
        "Value6": random.choices(range(100, 1000), k=num_rows),
    }
    return pd.DataFrame(data)


# Generate a DataFrame with 1000 rows of data
df = generate_data(1000)

# Save to CSV
df.to_csv("./data.csv", index=False)

df
