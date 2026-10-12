from setuptools import find_packages, setup
from typing import List
"just updated awsthings"
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='Fault detection',
    version='0.0.1',
    author='Sahil',
    author_email='sahilharadwala123@gmail.com',  
    install_requires=get_requirements('requirements.txt'),  
    packages=find_packages() 
)



"name=sahilharadwala123_db_user"
"password=C1DTgT5sznORnxzA"
"mongo ulr_= mongodb+srv://sahilharadwala123_db_user:C1DTgT5sznORnxzA@sensorfault.oz0wme4.mongodb.net/?appName=Sensorfault"