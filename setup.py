import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as file:
        return file.read()


setup(
    name="MonitoringCustomMetrics",
    version="1.0.4",
    description="Custom Metrics for ML Model Monitoring",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/amzn/MonitoringCustomMetrics",
    keywords="MonitoringCustomMetrics ML Monitoring Metrics",
    license="Apache License 2.0",
    packages=find_packages(where="src", exclude=("test",)),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved",
    ],
    package_dir={"": "src"},
)
