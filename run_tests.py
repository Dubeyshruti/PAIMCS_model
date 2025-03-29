import os
import sys
import unittest

# The project root is the directory containing both 'paimcs' and 'test_paimcs'
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)  # Now 'paimcs' is importable as a package

# Define the test directory (using its absolute path)
test_dir = os.path.join(project_root, 'test_paimcs')

# Discover and run all tests matching the pattern 'test_*.py' in test_paimcs/
loader = unittest.TestLoader()
suite = loader.discover(start_dir=test_dir, pattern='test_*.py')

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Exit with non-zero status if tests failed
sys.exit(not result.wasSuccessful())