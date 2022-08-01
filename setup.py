import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="phenoKOR", # 패키지 이름
        version="0.0.1", # 버전
        author="KNPS_GreenDay",
        author_email="gibum1228@gmail.com",
        description="phenology package project in KNPS",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/gibum1228/knps_phenology",
        project_urls={
            "issue": "https://github.com/gibum1228/knps_phenology/issues",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ],
        license="LICENSE",
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        python_requires=">=3.7",
    )