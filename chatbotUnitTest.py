#import libraries
import unittest
import chatBot

class test(unittest.TestCase):

# greetings test
    def test_greeting(self):
        self.assertIsInstance(chatBot.greeting("hi"), str)

# Basic_Q_1 test
    def test_Basic_Q_1(self):
        self.assertIsInstance(chatBot.basic1("help me"), str)

# Basic_Q_2 test
    def test_Basic_Q_2(self):
        self.assertIsInstance(chatBot.basic2("ok"), str)

# Basic_Q_3 test
    def test_Basic_Q_3(self):
        self.assertIsInstance(chatBot.basic3("Nop"), str)

# Basic_Q_4 test
    def test_Basic_Q_4(self):
        self.assertIsInstance(chatBot.basic4("Yes"), str)

# PlantMedicBot Response test
    def test_PlantMedicBot(self):
        self.assertIsInstance(chatBot.response("Kernel smut paddy disease"), str)


if __name__ == "__main__":
    unittest.main()
