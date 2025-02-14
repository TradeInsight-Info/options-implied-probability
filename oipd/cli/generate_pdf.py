from typing import Optional

from pandas import DataFrame
from traitlets import Bool

from oipd.core import calculate_pdf_and_cdf
from oipd.io import CSVReader


def run(
    input_csv_path: str,
    current_price: float,
    days_forward: int,
    risk_free_rate: float,
    fit_kernel_pdf: Optional[Bool] = False,
    save_to_csv: Optional[Bool] = False,
    output_csv_path: Optional[str] = None,
    solver_method: Optional[str] = "brent",
) -> DataFrame:
    """
    Runs the OIPD price distribution estimation using option market data.

    This function reads option data from a CSV file, calculates an implied probability
    density function (PDF) based on market prices, and optionally smooths the PDF
    using Kernel Density Estimation (KDE). It then computes the cumulative distribution
    function (CDF) and saves or returns the results.

    Args:
        input_csv_path (str): Path to the input CSV file containing option market data.
        current_price (float): The current price of the underlying security.
        days_forward (int): The number of days in the future for which the probability
            density is estimated.
        risk_free_rate (float): the annual risk free rate in nominal terms
        fit_kernel_pdf (Optional[bool], default=True): Whether to smooth the implied
            PDF using Kernel Density Estimation (KDE).
        save_to_csv (bool, default=False): If `True`, saves the output to a CSV file.
        output_csv_path (Optional[str], default=None): Path to save the output CSV file.
            Required if `save_to_csv=True`.
        solver_method (str): which solver to use for IV. Either "newton" or "brent"

    Returns:
        - If `save_to_csv` is `True`, saves the results to a CSV file and returns `None`.
        - If `save_to_csv` is `False`, returns a `pd.DataFrame` containing three columns:
          `Price`, `PDF`, and `CDF`.
    """

    reader = CSVReader()
    options_data = reader.read(input_csv_path)

    # Convert results to DataFrame
    df = calculate_pdf_and_cdf(
        options_data,
        current_price,
        days_forward,
        risk_free_rate,
        solver_method,
        fit_kernel_pdf,
    )

    # Save or return DataFrame
    if save_to_csv:
        if output_csv_path is None:
            raise ValueError("output_csv_path must be provided when save_to_csv=True")
        df.to_csv(output_csv_path, index=False)
        return df
    else:
        return df
