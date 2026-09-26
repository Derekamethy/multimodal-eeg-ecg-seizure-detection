import json

from src.multimodal_seizure.inventory import build_inventory


if __name__ == "__main__":
    manifest = build_inventory()
    print(json.dumps(manifest["inventory"], indent=2))
    print("METADATA_IDENTITY", manifest["metadata_identity"])
    print("ALL_SELECTED_EDF_MAPPINGS_VALIDATED")
