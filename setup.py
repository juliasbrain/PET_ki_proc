from setuptools import setup, find_packages

setup(
    name='pet_ki_proc',
    version='0.1.0',
    packages=find_packages(include=['pet_ki_proc', 'pet_ki_proc.*']),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    include_package_data=True,
    package_data={
        'pet_ki_proc': ['app/data/atlas/**/*'],
    },
    entry_points={
        'console_scripts': [
            'run_PET_ki_proc = pet_ki_proc.run_PET_ki_proc:main',
            'run_all = pet_ki_proc.run_all:main',
        ],
    },
    author="Julia Schulz",
    description="18F-DOPA PET data processing and ki estimation",
    python_requires=">=3.11",
    url="https://github.com/juliasbrain/PET",
)