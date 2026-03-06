from setuptools import find_packages, setup
from glob import glob

package_name = 'example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch*')),
        ('share/' + package_name + '/maps', ['example/Competition_track_graph.graphml']),
    ],
    install_requires=['setuptools', 'networkx'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'control = example.control:main',
            'camera = example.camera:main',
            'sign_detector = example.sign_detector:main',
            'reset_car = example.reset_car:main',
            'car_monitor = example.car_monitor:main',
            'navigator = example.navigator:main',
        ],
    },
)
