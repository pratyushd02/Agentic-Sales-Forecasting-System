import os
from typing import List, Dict, Any

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from mcp.server.fastmcp import FastMCP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

mcp = FastMCP("SalesForecastMCP")


def _dataset_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def _model_path(model_id: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_id}.joblib")


@mcp.tool()
def list_datasets() -> List[str]:
    """Return list of available CSV datasets in the datasets/ folder."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]


@mcp.tool()
def train_sales_model(
    dataset_name: str,
    date_column: str = "date",
    target_column: str = "sales",
) -> Dict[str, Any]:
    """
    Train a simple linear regression model on a time-indexed sales dataset.

    Args:
        dataset_name: CSV file in datasets/ (e.g. "sample_sales.csv")
        date_column: name of the date column
        target_column: name of the target (e.g. sales)

    Returns:
        {"model_id": "<id>", "rows_used": <int>}
    """
    path = _dataset_path(dataset_name)
    if not os.path.exists(path):
        raise ValueError(f"Dataset {dataset_name} not found in datasets/")

    df = pd.read_csv(path)
    if date_column not in df.columns or target_column not in df.columns:
        raise ValueError(
            f"Columns {date_column} or {target_column} not found in dataset."
        )

    # Sort by date and create a simple time index
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    df["t"] = range(1, len(df) + 1)

    X = df[["t"]]
    y = df[target_column]

    model = LinearRegression()
    model.fit(X, y)

    # Use dataset stem as model_id
    model_id = os.path.splitext(dataset_name)[0]
    joblib.dump(
        {
            "model": model,
            "last_t": int(df["t"].max()),
            "last_date": df[date_column].max(),
            "target_column": target_column,
        },
        _model_path(model_id),
    )

    return {"model_id": model_id, "rows_used": len(df)}


@mcp.tool()
def forecast_sales(
    model_id: str,
    horizon_months: int = 6,
) -> Dict[str, object]:
    """
    Use a trained model to forecast future sales over the next N time steps.

    Args:
        model_id: ID returned by train_sales_model
        horizon_months: number of future steps to forecast

    Returns:
        {
          "model_id": str,
          "horizon": int,
          "forecast": [float, ...]
        }
    """
    model_file = _model_path(model_id)
    if not os.path.exists(model_file):
        raise ValueError(f"Model {model_id} not found. Train it first.")

    data = joblib.load(model_file)
    model = data["model"]
    last_t = data["last_t"]

    future_t = [[last_t + i] for i in range(1, horizon_months + 1)]
    preds = model.predict(future_t)

    return {
        "model_id": model_id,
        "horizon": horizon_months,
        "forecast": [float(p) for p in preds],
    }


if __name__ == "__main__":
    # Run as stdio MCP server
    mcp.run()
