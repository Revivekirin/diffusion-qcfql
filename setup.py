from setuptools import setup, find_packages

setup(
    name="diffusion-qcfql",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchrl",
        "gymnasium",
        "numpy",
        "tqdm",
        "wandb",
        "ml-collections",
        "imageio",
        "h5py",
        "robosuite",
        # "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git",
    ],
    python_requires=">=3.8",
)
