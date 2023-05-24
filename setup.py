import re
import io

from setuptools import setup, find_packages


def get_version():
    __version__ = re.search(
        r'__VERSION__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open('sacreeos/__init__.py', encoding='utf_8_sig').read()
    ).group(1)
    return __version__


def get_description():
    __version__ = re.search(
        r'__DESCRIPTION__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open('sacreeos/__init__.py', encoding='utf_8_sig').read()
    ).group(1)
    return __version__


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()
    return long_description


setup(
    name='sacreeos',
    version=get_version(),
    description=get_description(),
    long_description_content_type='text/markdown',
    long_description=get_long_description(),
    url='https://github.com/jchenghu/sacreeos',
    author='Jia Cheng Hu',
    author_email='jia.jiachenghu@gmail.com',
    license='Apache License 2.0',
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Apache Software License',

        'Operating System :: POSIX',

        'Programming Language :: C++',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=['self-critical sequence training, computer vision, image captioning, evaluation'],
    packages=find_packages(),

    install_requires=[
        'torch',
        'numpy',
        'typing'
    ],

    entry_points={
        'console_scripts': [
          'sacreeos = sacreeos.sacreeos:main'
        ],
    }
)
