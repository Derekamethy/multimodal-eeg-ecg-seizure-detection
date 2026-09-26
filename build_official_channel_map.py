"""Compatibility entry point: rebuild the same canonical inventory, never a parallel map."""
from src.multimodal_seizure.inventory import build_inventory


if __name__ == "__main__":
    manifest = build_inventory()
    print("ALL_SELECTED_EDF_MAPPINGS_VALIDATED", manifest["inventory"]["selected_edfs"])
    print("METADATA_IDENTITY", manifest["metadata_identity"])
