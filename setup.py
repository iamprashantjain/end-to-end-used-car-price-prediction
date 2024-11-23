from setuptools import find_packages, setup
from typing import List

# Development mode package identifier
hyphen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements from the given file.
    
    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of package requirements.
    """
    requirements = []
    try:
        with open(file_path, 'r') as file_obj:
            # Read lines and strip whitespace
            requirements = [req.strip() for req in file_obj if req.strip()]
            
            # Remove the development mode package if it exists
            if hyphen_e_dot in requirements:
                requirements.remove(hyphen_e_dot)

    except FileNotFoundError:
        print(f"Warning: The requirements file {file_path} was not found.")
    except IOError as e:
        print(f"Error reading {file_path}: {e}")

    return requirements

# Setup configuration
setup(
    name="UsedCars-price-prediction",
    version="0.0.1",
    author='Prashant J',
    author_email='p@p.com',
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=get_requirements('requirements_dev.txt'),  # Get package requirements
)