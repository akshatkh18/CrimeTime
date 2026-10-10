"""Run with python -B -m unittest discover -s tests -v; never trains models."""

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    'data/cleaned.csv',
    'data/eda_stats.json',
    'model/rf_model.pkl',
    'model/encoders.pkl',
    'model/metrics.json',
)


class DeploymentSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ARTIFACTS if not (ROOT / name).is_file()]
        if missing:
            raise AssertionError(f'Missing runtime artifacts: {missing}')
        sys.path.insert(0, str(ROOT))
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                try:
                    cls.module = importlib.import_module('app')
                finally:
                    os.chdir(original_cwd)
        finally:
            sys.path.pop(0)
        cls.client = cls.module.app.test_client()

    def test_artifacts_and_import(self):
        self.assertEqual(self.module.BASE, ROOT)
        self.assertFalse(self.module.app.debug)
        self.assertGreater(len(self.module.df), 0)
        self.assertIsNotNone(self.module.model)
        self.assertIn('features', self.module.encoders)

    def test_all_five_routes(self):
        row = self.module.df.iloc[0]
        for path in ('/', '/eda'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'<html', response.data.lower())
        response = self.client.get('/api/eda_data')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), self.module.eda_stats)
        response = self.client.get('/get_districts', query_string={'state': row['STATE/UT']})
        self.assertEqual(response.status_code, 200)
        self.assertIn(row['DISTRICT'], response.get_json())
        response = self.client.post('/analyze', json={
            'state': row['STATE/UT'], 'district': row['DISTRICT'], 'year': int(row['YEAR']),
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['year'], int(row['YEAR']))
        self.assertEqual(result['safety_score'], round(float(row['SAFETY_SCORE']), 2))
        self.assertEqual(result['safety_label'], row['SAFETY_LABEL'])
        self.assertEqual(result['total_crimes'], int(row['TOTAL IPC CRIMES']))


if __name__ == '__main__':
    unittest.main()
