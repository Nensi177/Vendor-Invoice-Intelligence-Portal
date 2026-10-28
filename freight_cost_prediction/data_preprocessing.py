import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "inventory.db"


def load_vendor_invoice_data(db_path=None):
    """
    Load vendor invoice data from SQLite database.
    Defaults to data/inventory.db relative to the project root.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database file not found at: {db_path.resolve()}\n"
            f"Please ensure inventory.db exists under the data/ directory."
        )

    conn = sqlite3.connect(str(db_path))
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
