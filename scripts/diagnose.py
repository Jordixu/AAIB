"""Phase 7: read cv_summary, classify path A/B/C, write operating_envelope.json."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvasir_triage.eval.diagnose import run_diagnosis
from kvasir_triage.data.labels import EXCLUDED_CLASSES
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cv_summary_dir", type=Path)
    args = parser.parse_args()

    log = setup_logging("diagnose", logs_root())

    result = run_diagnosis(args.cv_summary_dir, excluded_classes=list(EXCLUDED_CLASSES))

    path = result["phase_7_path"]
    descriptions = {
        "A": "Success - model performs well and attends to the right image regions.",
        "B": "Clever Hans - performance looks good but CAM activations fall outside lesion area.",
        "C": "Data-bound - performance below threshold, likely due to dataset or training limitations.",
    }
    log.info("Phase 7 result: PATH %s", path)
    log.info("%s", descriptions[path])
    log.info("Operating envelope written to %s", args.cv_summary_dir / "operating_envelope.json")


if __name__ == "__main__":
    main()
