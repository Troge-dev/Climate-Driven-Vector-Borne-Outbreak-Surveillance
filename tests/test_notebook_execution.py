"""
Integration Test: Headless execution of all Jupyter Notebook code cells
"""

import unittest
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated testing
import matplotlib.pyplot as plt
from pathlib import Path

class TestNotebookExecution(unittest.TestCase):
    def test_notebooks_execution(self):
        for nb_rel_path in [
            "notebooks/02_synthetic_validation_and_stress_testing.ipynb",
            "notebooks/01_cchain_dengue_surveillance_pipeline.ipynb"
        ]:
            nb_path = Path(nb_rel_path)
            self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")

            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
            self.assertGreater(len(code_cells), 0, f"No code cells found in {nb_path}")

            env = {"__name__": "__main__"}
            def mock_display(obj):
                pass
            env["display"] = mock_display

            for idx, cell in enumerate(code_cells, 1):
                code_str = "".join(cell["source"])
                # Ensure headless plot closure
                code_str = code_str.replace("plt.show()", "plt.close('all')")
                try:
                    exec(code_str, env)
                except Exception as e:
                    self.fail(f"Execution failed in {nb_path.name} [Cell {idx}]: {e}")

if __name__ == "__main__":
    unittest.main()
