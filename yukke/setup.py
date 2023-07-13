import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="inkdet",
        version="1.2.1",
        py_modules=["inkdet"],
        author="yukke42",
        entry_points={
            "console_scripts": [
                "train = inkdet.tools.train:main",
                "train_classifier = inkdet.tools.train_classifier:main",
                "evaluate = inkdet.tools.evaluate:main",
            ],
        },
        packages=setuptools.find_packages(),
    )
