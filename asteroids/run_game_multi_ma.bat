@ECHO OFF

@set python=D:\Anaconda3\

@set to_run=main_xmars.py

REM ~ start "" /D . /AFFINITY 0x8  "%python%python" %to_run% --port 8888 --render 1
REM ~ start "" /D . /AFFINITY 0x7  "%python%python" %to_run% --port 8889
REM ~ start "" /D . /AFFINITY 0x6  "%python%python" %to_run% --port 8890
REM ~ start "" /D . /AFFINITY 0x5  "%python%python" %to_run% --port 8891
REM ~ start "" /D . /AFFINITY 0x4  "%python%python" %to_run% --port 8892
REM ~ start "" /D . /AFFINITY 0x3  "%python%python" %to_run% --port 8893
REM ~ start "" /D . /AFFINITY 0x2  "%python%python" %to_run% --port 8894
REM ~ start "" /D . /AFFINITY 0x1  "%python%python" %to_run% --port 8895

start "" /D . "%python%python" %to_run% --port 8872 --render 1
start "" /D . "%python%python" %to_run% --port 8873
start "" /D . "%python%python" %to_run% --port 8874
start "" /D . "%python%python" %to_run% --port 8875
start "" /D . "%python%python" %to_run% --port 8876
start "" /D . "%python%python" %to_run% --port 8877
start "" /D . "%python%python" %to_run% --port 8878
start "" /D . "%python%python" %to_run% --port 8879

start "" /D . "%python%python" %to_run% --port 8880
start "" /D . "%python%python" %to_run% --port 8881
start "" /D . "%python%python" %to_run% --port 8882
start "" /D . "%python%python" %to_run% --port 8883
start "" /D . "%python%python" %to_run% --port 8884
start "" /D . "%python%python" %to_run% --port 8885
start "" /D . "%python%python" %to_run% --port 8886
start "" /D . "%python%python" %to_run% --port 8887

REM ~ start "" /D . "%python%python" %to_run% --port 8888
REM ~ start "" /D . "%python%python" %to_run% --port 8889
REM ~ start "" /D . "%python%python" %to_run% --port 8890
REM ~ start "" /D . "%python%python" %to_run% --port 8891
REM ~ start "" /D . "%python%python" %to_run% --port 8892
REM ~ start "" /D . "%python%python" %to_run% --port 8893
REM ~ start "" /D . "%python%python" %to_run% --port 8894
REM ~ start "" /D . "%python%python" %to_run% --port 8895


REM ~ pause