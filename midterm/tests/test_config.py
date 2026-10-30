import unittest

from midterm.src.pipeline import cfg_for_method


class TestMethodConfig(unittest.TestCase):
    def test_section_wrapped_params_are_not_double_wrapped(self):
        base = {"association": {"min_support": 5, "top_k": 100}}
        params = {"association": {"min_support": 10, "top_k": 100}}
        out = cfg_for_method(base, "association", params)
        self.assertEqual(out["association"]["min_support"], 10)
        self.assertNotIn("association", out["association"])

    def test_flat_params_are_wrapped(self):
        base = {"repeat": {}}
        out = cfg_for_method(base, "repeat", {"half_life_days": 14})
        self.assertEqual(out["repeat"]["half_life_days"], 14)
        self.assertNotIn("repeat", out["repeat"])


if __name__ == "__main__":
    unittest.main()