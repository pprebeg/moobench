from setuptools import setup, find_packages

setup(
    name='moobench',
    version='0.1',
    packages=[''],
    package_dir={'': 'src'},
    install_requires=[
    'jmetalpy',
    'pymoo==0.5.0',
    'scipy'],
    url='https://github.com/pprebeg/moobench',
    license='MIT',
    author='Pero Prebeg',
    author_email='pero.prebeg@fsb.hr',
    description='Benchmark problems for multiobjective optimization with constraints',
)