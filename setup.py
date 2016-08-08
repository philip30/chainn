from setuptools import setup, find_packages
import sys

sys.path.append('./chainn')
sys.path.append('./test')

install_requires = [
    'Chainer>=1.10',
    'numpy>=1.9.0']

setup(
        name='Chainn',
        version='1.0.0',
        description='Neural Network Translation toolkit decoder',
        long_description='Chainn is a NMT Translation toolkit based on Chainer\
            It supports a basic encoder-decoder and attentional NMT models.\
            It also supports language modeling and pos tagging.',
        author='Philip Arthur',
        author_email='philip.arthur30@gmail.com',
        license='MIT License',
        install_requires=install_requires,
        packages=[
            'chainn',
            'chainn.util',
            'chainn.model',
            'chainn.machine',
            'chainn.util',
            'chainn.util.output',
            'chainn.util.io',
            'chainn.chainer_component',
            'chainn.chainer_component.functions',
            'chainn.chainer_component.links',
            'chainn.classifier',
            'chainn.model.basic',
            'chainn.model.nmt',
            'chainn.model.text',
            'chainn.test'
        ],
        platforms='requires Python Chainer Numpy'
        )
