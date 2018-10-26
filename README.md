# Machine-Learning-DQN

## Install requirements

```
pip install -r requirements.txt
```

## Install tensorflow

```
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.9.0-py3-none-any.whl
```

Might have to go to:
.virtualenvs/dq_testing/lib/python3.7/site-packages/tensorflow/python/pywrap_tensorflow_internal.py

Change keyword async in methods. (async is a keyword in python 3.6 and up) or 3.7

## Install game environments

```
pip install 'gym[all]'
```
