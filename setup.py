from setuptools import setup, find_packages

setup(
    name="stable",
    version="0.1.0",
    description="Pip package for the STABLE deep learning translation model",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "scikit-image",
        "tensorboard"
    ],
    entry_points={
        'console_scripts': [
            'stable_train=stable.train:main',
            'stable_infer=stable.infer:main'
        ]
    },
    python_requires='>=3.8'
)
