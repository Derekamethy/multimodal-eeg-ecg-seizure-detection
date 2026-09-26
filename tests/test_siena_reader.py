from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from multimodal_seizure.edf import read_channel_segment, read_header
from multimodal_seizure.siena import build_channel_map

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "siena-scalp-eeg-1.0.0"
PATIENTS = ("PN00", "PN06", "PN10", "PN12", "PN14")


@unittest.skipUnless(ROOT.exists(), "Siena raw subset is not available")
class TestSienaRealSubset(unittest.TestCase):
    def test_all_headers_and_channel_maps(self):
        total = 0
        for patient in PATIENTS:
            seizure_list = ROOT / patient / f"Seizures-list-{patient}.txt"
            for edf in sorted((ROOT / patient).glob("*.edf")):
                header = read_header(edf)
                mapping = build_channel_map(header, seizure_list)
                expected = 19 if patient == "PN10" else 29
                self.assertEqual(len(mapping.eeg_indices), expected)
                self.assertEqual(len(set(mapping.eeg_indices)), expected)
                self.assertEqual(len(mapping.ecg_indices), 2)
                self.assertTrue(all(abs(header.sampling_rate(i) - 512.0) < 1e-9 for i in range(header.n_signals)))
                self.assertEqual(tuple(header.signals[i].label for i in mapping.ecg_indices), ("1", "2"))
                total += 1
        self.assertEqual(total, 23)

    def test_read_two_seconds_per_patient(self):
        for patient in PATIENTS:
            edf = sorted((ROOT / patient).glob("*.edf"))[0]
            seizure_list = ROOT / patient / f"Seizures-list-{patient}.txt"
            header = read_header(edf)
            mapping = build_channel_map(header, seizure_list)
            selected = (*mapping.eeg_indices[:2], *mapping.ecg_indices)
            _, data = read_channel_segment(edf, selected, start_s=10.0, duration_s=2.0)
            for index in selected:
                self.assertEqual(data[index].shape, (1024,))
                self.assertTrue(np.isfinite(data[index]).all())
                self.assertGreater(float(np.std(data[index])), 0.0)


if __name__ == "__main__":
    unittest.main()
