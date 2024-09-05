import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestModelTrainning(unittest.TestCase):
    def test_model_trainning(self):
        model = joblib.load('model/model.pkl')
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()
