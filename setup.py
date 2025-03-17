from setuptools import setup, find_packages

setup(
    name="levee_hunter",
    version="2.0",
    packages=find_packages(),  # Auto-detects packages inside "levee_hunter/"
    author="Pawel Mucha, Marco Barnfield, Lioshan Shi",
    author_email="pawel_mucha@zohomail.eu",
    description="Hunting for levees using Lidar data",
    # url="https://github.com/pmucha/levee_hunter",  # Uncomment for GitHub link
    # license="MIT",  # Recommended to specify the license
    # classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    # ],
    python_requires=">=3.6",
    # install_requires=[
    #    # Add dependencies here, e.g., "numpy>=1.18.0", "rioxarray", "geopandas"
    # ],
    entry_points={
        "console_scripts": [
            "process-raw=levee_hunter.processing.process_raw_tifs:main",
        ]
    },
)
