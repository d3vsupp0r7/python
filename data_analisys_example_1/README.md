# Installation

## Install jupiterlab

1. 
Upgrade the pip package management

```
python -m pip install --upgrade pip
```

2.

Upgrade the setuptools

```
pip install --upgrade setuptools
```

3.

Install the **jupyterlab** tool

```
pip install jupyterlab
```

4.

Run the **jupyter notebook** tool

```
jupyter notebook
```

### Important notes for installation (05/12/2019)

```
 File "c:\users\<your_home_path>\appdata\local\programs\python\python38-32\lib\site-packages\tornado\platform\asyncio.py", line 99, in add_handler
    self.asyncio_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
  File "c:\users\<your_home_path>\appdata\local\programs\python\python38-32\lib\asyncio\events.py", line 501, in add_reader
    raise NotImplementedError
NotImplementedError
```

You need to modify the file **asyncio.py**, you need to insert the following lines of code after the 
**import asyncio** line:

```
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```