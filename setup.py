from setuptools import setup, find_packages

setup(
    name='ovrlpy',
    version='0.1.0',
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scanpy',
        'scikit-learn',
        'pandas',
    ],
    author='Sebastian Tiesmeyer',
    author_email='sebastian.tiesmeyer@bih-charite.de',
    description='A quality control tool for spatial transcriptomics data',
    url='https://github.com/HiDiH/ovrlpy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6,<3.13',
)