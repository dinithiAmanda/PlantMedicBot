#import libraries
import unittest
import app


class test(unittest.TestCase):

# Response test
    def test_PlantMediBot(self):
        self.assertIsInstance(app.PlantMedicBot("Kernel smut paddy disease"), str)

if __name__ == "__main__":
    unittest.main()