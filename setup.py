from setuptools import setup, find_packages

setup(
    name="MultiscaleBlackBox",
    version="0.1",
    description="Contains tools to create multiscale structures and optimize them",
    author="Artem",
    author_email="artyomchuprow@yandex.ru",
    packages=["blackbox_optimizer", "multilayer_tools"],
    install_requires=[
        "numpy>=2.0",
        "mpi4py>=4.0",
    ],
)

