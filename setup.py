from setuptools import setup

package_name = 'scuba_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='faraz',
    maintainer_email='farazlotfi1992@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['gathering_data = scuba_tracking.gathering_dataset:main',
        'vision_module = scuba_tracking.object_tracker:main',
        'controller = scuba_tracking.classic_controller:main'
        ],
    },
)
