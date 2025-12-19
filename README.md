This project uses some codes from https://github.com/Learning-and-Intelligent-Systems/proc3s regarding LLM parse utils.  

The main part of code is based on Carla's open source example code manual_control.py  

# Installation  

To install, first download Carla Simulator 0.9.16 or later, https://carla.readthedocs.io/en/latest/start_quickstart/#download-and-extract-a-carla-package  

Then cover PythonAPI with the content here.  

Finally run manual_control.py  

# How To Use  
Press W, A, S, D to move  
Press blackspace to change vehicle  
When reach a starting point, press M to send query to LLM, then you can look at the terminal and wait for response to complete  
After completion, press KP1, KP2, KP3 or K, Y, N to select route 1, 2 or 3  
Press Z to preview LLM final path
Press X to preview A* with exclusive area search final path
Press C to actually move along LLM path
