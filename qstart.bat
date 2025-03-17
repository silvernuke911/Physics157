@echo off
:: ==============  QUICKSTART  ==============
:: > C:\Users\verci\qstart.bat
:: ==========================================
:: Script which opens the things I want open 
:: after startup immediately. 

:: Makes cmd color green (personal preference)
color 0a

:: Open Notepad 
start /min "" notepad                  

:: Open Chrome profiles 
start /min "" chrome --profile-directory="Default"   
start /min "" chrome --profile-directory="Profile 1" 

:: Start VS Code 
code

:: Open File Explorer to Documents 
start "" explorer "C:\Users\verci\Documents"
timeout /t 2 /nobreak >nul

:: Standard startup for MS products (uncomment if needed)
:: start "" winword
:: start "" powerpnt
:: start "" excel

:: Open Windows Settings (uncomment if needed)
:: start "" ms-settings:

:: For system cleaning
:: SFC

:: Shutdown command (DO NOT USE)
:: shutdown /s /t 10
