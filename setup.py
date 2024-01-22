from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [
                s.strip()
                for s in f.readlines()
                if (s.strip() and not s.startswith("#"))
            ]

    REQUIREMENTS = _read_reqs("requirements.txt")
    REQUIREMENTS_BLENDER = _read_reqs("requirements-blender.txt")
    REQUIREMENTS_ANNOTATION = _read_reqs("requirements-annotation.txt")

    setup(
        name="objathor",
        packages=find_packages(),
        include_package_data=True,
        version="1.0.0",
        license="MIT",
        description="Objaverse asset importer for THOR",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Allen Institute for AI",
        author_email="contact@allenai.org",
        url="https://github.com/allenai/objathor",
        data_files=[(".", ["README.md"])],
        keywords=["3D assets", "annotation", ""],
        install_requires=REQUIREMENTS,
        extras_require={
            "blender": REQUIREMENTS_BLENDER,
            "annotation": REQUIREMENTS_ANNOTATION,
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.8",
        ],
    )
