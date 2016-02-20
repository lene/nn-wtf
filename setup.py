from setuptools import setup

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='nn_wtf',
    version='0.1.1',
    description='Neural Networks Wrapper for TensorFlow',
    long_description='Really, the funniest around.',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning neural networks tensorflow',
    url='http://github.com/lene/nn-wtf',
    author='Lene Preuss',
    author_email='lene.preuss@gmail.com',
    license='Apache Software License V2',
    packages=['nn_wtf'],
    install_requires=[
        # 'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl',
        'numpy'
    ],
    include_package_data=True,
    zip_safe=False
)
