#!/usr/bin/env python3
"""
SudenMind 安装脚本
基于 ChatGLM tokenizer + 自定义 AttnResEncoder 的中文对话生成模型
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="sudenmind",
    version="3.0.0",
    author="SudenMind Team",
    author_email="",
    description="Chinese dialogue generation model with custom AttnResEncoder/Decoder and ChatGLM tokenizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sudenmind-process=src.process:main",
            "sudenmind-train=src.train:main",
            "sudenmind-chat=src.chat:chat",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config.json", "data/raw/*.txt", "data/processed/*.json"],
    },
)
