from typing import Optional
from traitlets import Bool
from pandas import DataFrame
from .pdf import calculate_cdf, calculate_pdf, fit_kde


def calculate_pdf_and_cdf(
    options_data: DataFrame,
    current_price: float,
    days_forward: int,
    risk_free_rate: float = 0.03,
    solver_method: Optional[str] = "brent",
    fit_kernel_pdf: Optional[Bool] = False,
):
    pdf_point_arrays = calculate_pdf(
        options_data, current_price, days_forward, risk_free_rate, solver_method
    )

    # Fit KDE to normalize PDF if desired
    if fit_kernel_pdf:
        pdf_point_arrays = fit_kde(
            pdf_point_arrays
        )  # Ensure this returns a tuple of arrays

    cdf_point_arrays = calculate_cdf(pdf_point_arrays)  # type: ignore

    priceP, densityP = pdf_point_arrays  # type: ignore
    _, densityC = cdf_point_arrays

    # Convert results to DataFrame
    df = DataFrame({"Price": priceP, "PDF": densityP, "CDF": densityC})

    return df
