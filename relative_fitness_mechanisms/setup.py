import setuptools


def read_file(name):
    with open(name, "r") as f:
        return f.read()


setuptools.setup(
    name="relative_fitness_mechanisms",
    version="0.0.1",
    author="Marlin Figgins",
    author_email="marlinfiggins@gmail.com",
    description="Placeholder.",
    long_description=read_file("../README.md"),
    license="MIT",
    url="https://github.com/blab/relative-fitness-mechanisms",
    install_requires=[
        "matplotlib",
        "numpy",
        "jax",
        "pandas",
    ],
)
