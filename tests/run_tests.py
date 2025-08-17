import unittest

if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Discover tests in each subfolder
    for folder in ["graph_generation", "data", "llm"]:
        subfolder_tests = test_loader.discover(f"tests/{folder}")
        test_suite.addTests(subfolder_tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
