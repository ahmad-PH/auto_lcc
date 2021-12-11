import setuptools

setuptools.setup(
    name="iml_group_proj",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description="not long",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="./src", exclude=("./tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
