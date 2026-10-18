"""Dataset Download and Ingestion Module.

Fetches the raw Telco Customer Churn dataset from the verified public IBM repository
and stores it safely in data/raw/ without modification.

Dataset Provenance:
- Source: IBM Telco Customer Churn Sample Dataset
- Upstream URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Upstream Project: IBM Cloud Pak for Data Code Patterns
- License: Apache License 2.0 (Permissive for commercial, educational, and research use)
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default public repository URL hosted by IBM
DATASET_SOURCE_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# License and provenance metadata for transparency and interview readiness
DATASET_METADATA = {
    "name": "Telco Customer Churn",
    "source_organization": "IBM Developer",
    "upstream_repository": "https://github.com/IBM/telco-customer-churn-on-icp4d",
    "source_url": DATASET_SOURCE_URL,
    "license": "Apache License 2.0",
    "usage_terms": "Open for research, educational, and development purposes.",
}


def download_telco_churn_dataset(
    target_path: Optional[Path] = None,
    source_url: str = DATASET_SOURCE_URL,
    force_download: bool = False,
) -> Path:
    """Download the raw Telco Customer Churn CSV dataset to target_path.

    Parameters
    ----------
    target_path : Optional[Path], default None
        Destination path for the raw CSV. If None, defaults to `data/raw/Telco-Customer-Churn.csv`
        resolved relative to the project root.
    source_url : str, default DATASET_SOURCE_URL
        The verified public URL from which to download the dataset.
    force_download : bool, default False
        If True, re-downloads even if the file already exists locally.

    Returns
    -------
    Path
        The path to the downloaded raw dataset file.

    Raises
    ------
    RuntimeError
        If the download fails or the downloaded file is empty.
    """
    if target_path is None:
        # Resolve project root dynamically based on current module location:
        # src/data/download.py -> src/data -> src -> project_root
        project_root = Path(__file__).resolve().parents[2]
        target_path = project_root / "data" / "raw" / "Telco-Customer-Churn.csv"
    else:
        target_path = Path(target_path)

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force_download:
        file_size = target_path.stat().st_size
        if file_size > 0:
            logger.info(
                "Raw dataset already exists at %s (size: %d bytes). Skipping download. "
                "Use force_download=True to re-download.",
                target_path,
                file_size,
            )
            return target_path

    logger.info("Downloading raw dataset from verified source: %s", source_url)
    try:
        # Add a standard user-agent header to avoid potential rate-limiting or blocking
        req = urllib.request.Request(
            source_url,
            headers={"User-Agent": "CustomerChurnIntelligencePipeline/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
            data = response.read()
            if not data:
                raise RuntimeError("Received empty response from dataset source URL.")
            out_file.write(data)

        file_size = target_path.stat().st_size
        logger.info(
            "Successfully downloaded raw dataset to %s (size: %d bytes).",
            target_path,
            file_size,
        )
        return target_path

    except Exception as exc:
        # Clean up partial download if it exists
        if target_path.exists() and target_path.stat().st_size == 0:
            target_path.unlink(missing_ok=True)
        logger.error("Failed to download dataset from %s: %s", source_url, exc)
        raise RuntimeError(f"Could not download dataset: {exc}") from exc


if __name__ == "__main__":
    download_telco_churn_dataset()
