from setuptools import setup, find_packages

setup(
    name='pnpfit',
    url='https://github.com/annawoodard/pnpfit',
    author='Anna Woodard',
    author_email='annawoodard@uchicago.edu',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'parsl',
        'matplotlib'
    ],
    keywords=['Workflows', 'Scientific computing', 'High Energy Physics', 'Compact Muon Solenoid'],
)
