from pathlib import Path

import pandas as pd


# Assumptions for business impact calculation
COST_PER_DEFECTIVE_UNIT = 500
POTENTIAL_REDUCTION_RATE = 0.25


def load_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned production data."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


def calculate_business_impact(
    data: pd.DataFrame
) -> dict:
    """Calculate quality-related business impact."""

    total_units = len(data)

    defective_units = int(
        (data["quality_label"] == 1).sum()
    )

    good_units = int(
        (data["quality_label"] == 0).sum()
    )

    defect_rate = (
        defective_units / total_units
        if total_units > 0
        else 0
    )

    estimated_loss = (
        defective_units *
        COST_PER_DEFECTIVE_UNIT
    )

    potential_defect_reduction = (
        defective_units *
        POTENTIAL_REDUCTION_RATE
    )

    potential_savings = (
        potential_defect_reduction *
        COST_PER_DEFECTIVE_UNIT
    )

    return {
        "total_units": total_units,
        "good_units": good_units,
        "defective_units": defective_units,
        "defect_rate": defect_rate,
        "cost_per_defective_unit": COST_PER_DEFECTIVE_UNIT,
        "estimated_quality_loss": estimated_loss,
        "potential_defect_reduction": (
            potential_defect_reduction
        ),
        "potential_savings": potential_savings
    }


def save_results(
    results: dict,
    output_path: Path
) -> None:
    """Save business impact results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    data_path = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    output_path = (
        project_root
        / "outputs"
        / "reports"
        / "business_impact.csv"
    )

    print(
        "\n========== BUSINESS IMPACT ANALYSIS =========="
    )

    print("\nLoading production data...")

    data = load_data(
        data_path
    )

    results = calculate_business_impact(
        data
    )

    print(
        "\n========== BUSINESS IMPACT RESULTS =========="
    )

    print(
        f"Total production units: "
        f"{results['total_units']}"
    )

    print(
        f"Good units: "
        f"{results['good_units']}"
    )

    print(
        f"Defective units: "
        f"{results['defective_units']}"
    )

    print(
        f"Defect rate: "
        f"{results['defect_rate']:.2%}"
    )

    print(
        f"Cost per defective unit: "
        f"₹{results['cost_per_defective_unit']:,.2f}"
    )

    print(
        f"Estimated quality loss: "
        f"₹{results['estimated_quality_loss']:,.2f}"
    )

    print(
        f"Potential defect reduction: "
        f"{results['potential_defect_reduction']:.2f} units"
    )

    print(
        f"Potential savings: "
        f"₹{results['potential_savings']:,.2f}"
    )

    save_results(
        results,
        output_path
    )

    print(
        "\nBusiness impact report saved to:"
    )

    print(output_path)

    print(
        "\n========== BUSINESS IMPACT ANALYSIS COMPLETE =========="
    )