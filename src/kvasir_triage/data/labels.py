"""Label definitions and binary mapping."""

DIR_NAME_MAP: dict[str, str] = {
    "Ampulla of vater": "Ampulla of Vater",
    "ampulla_of_vater": "Ampulla of Vater",
    "angiectasia": "Angiectasia",
    "blood_fresh": "Blood - fresh",
    "blood_hematin": "Blood - hematin",
    "erosion": "Erosion",
    "erythema": "Erythema",
    "foreign_body": "Foreign body",
    "ileocecal_valve": "Ileocecal valve",
    "lymphangiectasia": "Lymphangiectasia",
    "normal_clean_mucosa": "Normal clean mucosa",
    "polyp": "Polyp",
    "pylorus": "Pylorus",
    "reduced_mucosal_view": "Reduced mucosal view",
    "ulcer": "Ulcer",
}

# All subclasses present in Kvasir-Capsule labelled_images
CANONICAL_LABELS = [
    "Angiectasia",
    "Blood - fresh",
    "Blood - hematin",
    "Erosion",
    "Erythema",
    "Foreign body",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced mucosal view",
    "Ulcer",
    "Ampulla of Vater",
]

# Excluded from project
EXCLUDED_CLASSES = {"Ampulla of Vater", "Blood - fresh", "Blood - hematin", "Polyp"}

# Subclasses that are anatomical landmarks (not pathology) -> Normal
NORMAL_CLASSES = {
    "Ileocecal valve",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced mucosal view",
}

# Everything else (non-excluded, non-normal) -> Suspicious
SUSPICIOUS_CLASSES = {
    c for c in CANONICAL_LABELS
    if c not in EXCLUDED_CLASSES and c not in NORMAL_CLASSES
}

# Map subclass name -> binary label (0=Normal, 1=Suspicious)
BINARY_LABEL: dict[str, int] = {
    **{c: 0 for c in NORMAL_CLASSES},
    **{c: 1 for c in SUSPICIOUS_CLASSES},
}


def subclass_to_binary(subclass: str) -> int:
    """Return 0 (Normal) or 1 (Suspicious). Raises KeyError for excluded/unknown."""
    return BINARY_LABEL[subclass]


def is_excluded(subclass: str) -> bool:
    return subclass in EXCLUDED_CLASSES
