# Notes

## Get Path of Current Python Intepreter
```
conda info --envs | grep '\*' | awk '{print $NF}'
```