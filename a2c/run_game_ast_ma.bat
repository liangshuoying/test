@ECHO OFF

REM ~ @set python=D:\Anaconda\

@set current_dir=..\asteroids
pushd %current_dir% 

@run_game_multi_ma.bat

popd

REM ~ pause