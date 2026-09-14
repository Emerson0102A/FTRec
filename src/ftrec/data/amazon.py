"""Amazon Reviews 2023 five-domain constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    name: str
    domain_id: int
    filename: str


AMAZON5_DOMAINS = (
    DomainSpec("Health", 0, "Health_and_Household.csv.gz"),
    DomainSpec("Clothing", 1, "Clothing_Shoes_and_Jewelry.csv.gz"),
    DomainSpec("Beauty", 2, "Beauty_and_Personal_Care.csv.gz"),
    DomainSpec("Grocery", 3, "Grocery_and_Gourmet_Food.csv.gz"),
    DomainSpec("Sports", 4, "Sports_and_Outdoors.csv.gz"),
)
DOMAIN_NAMES = tuple(domain.name for domain in AMAZON5_DOMAINS)
DOMAIN_BY_ID = {domain.domain_id: domain for domain in AMAZON5_DOMAINS}
DOMAIN_BY_NAME = {domain.name: domain for domain in AMAZON5_DOMAINS}

