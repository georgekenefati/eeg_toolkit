from setuptools import setup, find_packages
import numpy as np

setup(
    name="wanglab_eeg",
    description="blank",
    long_description='Wang Lab EEG Python Code'
                     'GitHub: https://github.com/georgekenefati/wanglab_eeg',
    version="0.1",
    python_requires='>=3.6',

    install_requires=[
        'mne', 'numpy', 'os',
    ],

    # metadata for upload to PyPI
    author="George Kenefati",
    author_email="george.kenefati@nyulangone.org",
    license="BSD 2-Clause (Simplified)",
    # cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    packages=find_packages(),
    # ext_modules=extensions,
    url='dummy-url',
    project_urls={
        "Source Code": '',
    }
)