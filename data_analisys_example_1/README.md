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

# Useful external reference

https://medium.com/diogo-menezes-borges/project-1-bigmart-sale-prediction-fdc04f07dc1e


## Data science

https://www.innoarchitech.com/blog/what-is-data-science-does-data-scientist-do

## Feature selection

https://medium.com/@mehulved1503/feature-selection-and-feature-extraction-in-machine-learning-an-overview-57891c595e96

https://medium.com/@mehulved1503/feature-selection-in-machine-learning-variable-ranking-and-feature-subset-selection-methods-89b2896f2220

https://towardsdatascience.com/feature-selection-techniques-1bfab5fe0784

https://www.machinelearningplus.com/machine-learning/feature-selection/

https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b

## Feature engineering

https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

## Outlier

https://medium.com/@swethalakshmanan14/outlier-detection-and-treatment-a-beginners-guide-c44af0699754

https://medium.com/@mehulved1503/effective-outlier-detection-techniques-in-machine-learning-ef609b6ade72

https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a

## Feature management

https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e

## Python

https://www.jeannicholashould.com/tidy-data-in-python.html

## Scrum-Agile context with ML

https://towardsdatascience.com/how-to-run-scrum-in-data-science-teams-56ddbe2ec8a5

https://medium.com/qash/how-and-why-to-use-agile-for-machine-learning-384b030e67b6

## ML examples

### a
https://medium.com/@powersteh/an-introduction-to-applied-machine-learning-with-multiple-linear-regression-and-python-925c1d97a02b

### b
https://towardsdatascience.com/a-data-science-case-study-with-python-mercari-price-prediction-4e852d95654

https://towardsdatascience.com/predicting-sales-611cb5a252de
