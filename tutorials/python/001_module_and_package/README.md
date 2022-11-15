## Structure of Package

Ref : https://realpython.com/python-modules-packages/#python-packages 

```
.
├── README.md
└── package
    ├── __init__.py
    ├── sub1
    │   ├── __init__.py
    │   ├── sub1_mod1.py
    │   └── sub1_mod2.py
    └── sub2
        ├── sub2_mod1.py
        └── sub2_mod2.py
```


### Try below codes

```python
import package
```

```python
import package.sub1
```

```python
import package.sub1.sub1_mod1
```
