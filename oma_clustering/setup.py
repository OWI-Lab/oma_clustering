#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'scikit-learn>=1.3.0']

test_requirements = ['pytest>=3', ]

setup(
    author="Maximillian Weil",
    author_email='maximillian.weil@vub.be',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python package to cluster the outputs of an Operational Modal Analysis. The clusters serve as a basis for automatically setting the configuration of further mode tracking.",
    entry_points={
        'console_scripts': [
            'oma_clustering=oma_clustering.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='oma_clustering',
    name='oma_clustering',
    packages=find_packages(include=['oma_clustering', 'oma_clustering.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/WEILMAX/oma_clustering',
    version='0.1.0',
    zip_safe=False,
)
