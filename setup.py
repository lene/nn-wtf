from setuptools import setup

try:
    from pypandoc import convert

    def read_md(f):
        return convert(f, 'rst')
except ImportError:
    def read_md(f):
        with open(f, 'r') as h:
            return h.read()


setup(
    name='nn_wtf',
    version='0.1.6',
    description='Neural Networks Wrapper for TensorFlow',
    long_description=read_md('README.md'),
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
    package_data={'nn_wtf': ['data/*.raw']},
    install_requires=[
        'numpy'
    ],
    dependency_links=[
        'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.0-py3-none-linux_x86_64.whl'
    ],
    include_package_data=True,
    zip_safe=False
)
