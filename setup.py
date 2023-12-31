import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
	name="servier",
    version="1.0.0",
    author="Hugo Todeschini",
    author_email="todeschinihugo@gmail.com",
    description="Package to train a machine learning model to predict P1 molecule properties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HugoTodeschini/molecule-properties-prediction.git",
    packages=setuptools.find_packages(where='src',),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'servier_train = model1:train',
            'servier_evaluate = model1:evaluate',
            'servier_predict = model1:predict',
        ],
    },
)