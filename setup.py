from setuptools import setup, find_packages

setup(
    name="binary_collision",
    version="1.0",
    packages=find_packages(include=["binary_collision", "binary_collision.*"]),
    install_requires=["numpy", "scipy", "matplotlib"],
    extras_require={
        "dev": [
            "numpy==1.23.5",
            "scipy==1.10.0",
            "matplotlib==3.7.0",
            "pytest==7.1.2",
        ]
    },
    python_requires=">=3.10",
    description="Plasma Binary Collision Operator Library",
    author="Sungpil YUM",
    author_email="sungpil.yum@unist.ac.kr",
    url="https://github.com/UNIST-FPL/binary-coulomb-collision",
)
