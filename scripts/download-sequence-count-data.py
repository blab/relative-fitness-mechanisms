# Importing necessary libraries
import os

import requests


def check_and_download_file(file_path, url):
    if os.path.exists(file_path):
        return "File already exists."
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        with open(file_path, "wb") as file:
            file.write(response.content)
        return "File downloaded successfully."
    except requests.exceptions.RequestException as e:
        return f"Error downloading file: {e}"


SCALES = ["global", "usa"]

PANGO_LINEAGE_PATH = "./pango-lineage/global.tsv.gz"
PANGO_LINEAGE_URL = "https://data.nextstrain.org/files/workflows/forecasts-ncov/gisaid/pango_lineages/global.tsv.gz"

NEXTSTRAIN_CLADE_PATH = "./nextstrain-clade/global.tsv.gz"
NEXTSTRAIN_CLADE_URL = "https://data.nextstrain.org/files/workflows/forecasts-ncov/gisaid/nextstrain_clades/global.tsv.gz"

NEXTSTRAIN_CLADE_PATH_USA = "./nextstrain-clade-usa/usa.tsv.gz"
NEXTSTRAIN_CLADE_URL_USA = "https://data.nextstrain.org/files/workflows/forecasts-ncov/gisaid/nextstrain_clades/usa.tsv.gz"

if __name__ == "__main__":
    check_and_download_file(PANGO_LINEAGE_PATH, PANGO_LINEAGE_URL)
    check_and_download_file(NEXTSTRAIN_CLADE_PATH, NEXTSTRAIN_CLADE_URL)
    check_and_download_file(NEXTSTRAIN_CLADE_PATH_USA, NEXTSTRAIN_CLADE_URL_USA)

