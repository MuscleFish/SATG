from setuptools import setup
import setuptools

packages = setuptools.find_packages()

setup(name='SATG',
         version='0.2',
         author='gwh',
         description='Seif Adversial Topic Generation',
         python_requires='>=3.5',
         install_requires=['tensorflow>=2.2.3', 'tabulate', 'tensorflow_addons', 'scikit-learn', 
                           'numpy', 'keras-tcn', 'joblib', 'matplotlib'],
         py_modules=['BaseLayers', 'SATG'],
         packages=packages,
         classifiers=[
             "Programming Language :: Python :: 3",
         ],
        )