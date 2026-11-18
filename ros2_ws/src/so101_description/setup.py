from glob import glob

from setuptools import setup

package_name = "so101_description"

setup(
    name=package_name,
    version="1.0.0",
    packages=["so101_description"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        (f"share/{package_name}/meshes", glob("meshes/*.stl")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    zip_safe=True,
    description="URDF SO-ARM101 + launch de visualisation (fourni).",
    license="Apache-2.0",
)
