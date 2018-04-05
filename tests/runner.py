import unittest

loader = unittest.TestLoader()
start_dir = "./tests"
suite = loader.discover(".")

runner = unittest.TextTestRunner()
runner.run(suite)