from setuptools import setup, find_packages

setup(
    name="Zepto_DS_Project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning model API with CI/CD pipeline",
    packages=find_packages(),
    install_requires=[
        "flask",
        "scikit-learn==1.6.1",
        "numpy",
    ],
)
