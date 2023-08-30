import pandas as pd
import json


def read_data(path: str = "data/crude_oil_prices.csv") -> pd.DataFrame:
    """ load your data here. This may look something like:
        with open(path) as f:
                data = json.load(f)
            return pd.DataFrame(data["records"])
    """
    return pd.read_csv(path)


print(read_data().tail())
