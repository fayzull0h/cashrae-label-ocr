"""Category-specific configuration for ENERGY STAR product snapshots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyMetric:
    field: str
    name: str
    unit: str


@dataclass(frozen=True)
class EnergyStarDataset:
    category: str
    dataset_id: str
    dataset_url: str
    brand_field: str = "brand_name"
    model_field: str = "model_number"
    record_id_field: str = "pd_id"
    annual_energy_field: str = "annual_energy_use_kwh_year"
    annual_energy_unit: str = "kWh/year"
    annual_energy_context: str | None = None
    metrics: tuple[EnergyMetric, ...] = ()

    @property
    def table(self) -> str:
        return f"energy_star_{self.category}"

    @property
    def api_url(self) -> str:
        return f"https://data.energystar.gov/resource/{self.dataset_id}.json"


ENERGY_STAR_DATASETS: dict[str, EnergyStarDataset] = {
    "clothes_washers": EnergyStarDataset(
        category="clothes_washers",
        dataset_id="bghd-e2wd",
        dataset_url=("https://data.energystar.gov/Active-Specifications/"
                     "ENERGY-STAR-Certified-Residential-Clothes-Washers/bghd-e2wd"),
        metrics=(EnergyMetric("integrated_modified_energy_factor_imef",
                              "IMEF",
                              "ft³/kWh/cycle"),),
    ),
    "room_air_conditioners": EnergyStarDataset(
        category="room_air_conditioners",
        dataset_id="5xn2-dv4h",
        dataset_url=("https://data.energystar.gov/Active-Specifications/"
                     "ENERGY-STAR-Certified-Room-Air-Conditioners/5xn2-dv4h"),
        annual_energy_field="annual_energy_use_kwh_yr",
        annual_energy_context=(
            "ENERGY STAR's estimated annual use assumes 750 compressor operating hours "
            "per year; actual consumption varies with use."
        ),
        metrics=(
            EnergyMetric("combined_energy_efficiency_ratio_ceer", "CEER", "Btu/Wh"),
            EnergyMetric("cooling_capacity_btu_hour", "Cooling capacity", "Btu/hr"),
            EnergyMetric("percent_less_energy_use_than_us_federal_standard",
                         "Less energy than U.S. federal standard", "%"),
        ),
    ),
    "refrigerators": EnergyStarDataset(
        category="refrigerators",
        dataset_id="p5st-her9",
        dataset_url="https://data.energystar.gov/w/p5st-her9/im48-wy4k",
        annual_energy_field="annual_energy_use_kwh_yr",
        metrics=(
            EnergyMetric("capacity_total_volume_ft3", "Capacity (total volume)", "ft³"),
            EnergyMetric("adjusted_volume_ft3", "Adjusted volume", "ft³"),
            EnergyMetric("us_federal_standard_kwh_yr", "U.S. federal standard", "kWh/year"),
            EnergyMetric("percent_less_energy_use_than_us_federal_standard",
                         "Less energy than U.S. federal standard", "%"),
        ),
    ),
    "dishwashers": EnergyStarDataset(
        category="dishwashers",
        dataset_id="q8py-6w3f",
        dataset_url=("https://data.energystar.gov/Active-Specifications/"
                     "ENERGY-STAR-Certified-Residential-Dishwashers/q8py-6w3f"),
        annual_energy_context=(
            "Estimated under typical conditions and includes dishwasher cycle energy, "
            "additional energy used by a water heater to heat the water, and standby "
            "energy. Based on 215 loads per year (about four per week) under the DOE "
            "test procedure; actual use varies with usage patterns and cycle selection."
        ),
        metrics=(
            EnergyMetric("water_use_gallons_cycle", "Water use", "gallons/cycle"),
            EnergyMetric("us_federal_standard_kwh_year", "U.S. federal energy standard", "kWh/year"),
            EnergyMetric("better_than_us_federal_standard_kwh_year",
                         "Energy use below U.S. federal standard", "%"),
            EnergyMetric("us_federal_standard_gallons_cycle", "U.S. federal water standard",
                         "gallons/cycle"),
            EnergyMetric("better_than_us_federal_standard_gallons_cycle",
                         "Water use below U.S. federal standard", "%"),
        ),
    ),
}

# Existing behavior treats a washer-dryer category and an unspecified category as
# washer lookups. Keep that mapping while enabling the new product categories.
ENERGY_STAR_CATEGORY_MAP = {
    None: "clothes_washers",
    "clothes_washers": "clothes_washers",
    "clothes_washer_dryers": "clothes_washers",
    "room_air_conditioners": "room_air_conditioners",
    "refrigerators": "refrigerators",
    "dishwashers": "dishwashers",
}


def dataset_for_category(category: str | None) -> EnergyStarDataset | None:
    dataset_category = ENERGY_STAR_CATEGORY_MAP.get(category)
    return ENERGY_STAR_DATASETS.get(dataset_category)
