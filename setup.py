from setuptools import setup, find_packages

setup(
    name="airom_mc",  # Replace with your project name
    version="0.1",  # Replace with your project version
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        # List your project's dependencies here, e.g.:
        # 'numpy>=1.18.0',
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here, e.g.:
            # 'your-script=your_module:main_function',
        ],
    },
    )
