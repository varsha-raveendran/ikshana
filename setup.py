"""Define setup for installing the repository as a pip package."""
from setuptools import find_packages, setup

setup(
    name="dlutils-eva6-group15",
    packages=find_packages(),
    version="0.1.1",
    description="Python package for computer vision",
    author="EVA6 Group 15",
    license="MIT",
    url="https://github.com/EVA6-Group-15/DLUtils",
    install_requires=[
        "click==7.1.2",
        "Sphinx==4.0.2",
        "-f https://download.pytorch.org/whl/torch_stable.html",
        "torch==1.8.1+cu111",
        "torchvision==0.9.1+cu111",
        "torchaudio===0.8.1 ",
        "matplotlib==3.4.2",
        "numpy==1.20.3",
        "pandas==1.2.4",
        "hiddenlayer==0.3",
        "seaborn==0.11.1",
        "torchsummary==1.5.1",
        "tqdm==4.61.0",
        "imgaug==0.4.0",
        "albumentations==1.0.0",
        "python-dotenv>=0.5.1",
    ],
    extras_require={
        "dev": [
            "black==21.6b0",
            "pylint==2.8.3",
            "pydocstyle==6.1.1",
            "mypy==0.902",
            "pre-commit==2.13.0",
            "isort==5.8.0",
            "jupyter==1.0.0",
            "notebook==6.4.0",
            "jupyterlab==3.0.16",
        ],
    },
)
