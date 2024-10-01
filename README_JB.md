This file contains 

config/my contains .yaml files I used in runs 

# CompareKotlin2 

It takes two version of kotlin compiler, folder with .fuzz files and compare results of .fuzz files on two compilers. Usage:


main_with_config --folder 'folder with .fuzz' --target_1 'path to kotlinc' --target_2 'path to kotlinc'

It will write all results to --folder/log_compare.txt

# KoverCoverage

This module build graph of the kotlin compiler koverage through iterations of fuzzer results.

Usage:

compiler source output size --scope=100 --restart=false

compiler - path to kotlinc
source - fuzz4All output folder with .fuzz files
output - output folder for 
size - number of .fizz files
--scope - frequency of measurments on graph
--restart - set true if you want to start algorithm from 0 iteration

You can restart your run in case of any error. it saves results after each iteration. Just run again with the --restart=true and it will use experiment.json from your output folder.

All coverage info will be located in output folder in experiment.json

# Fuzz4All

Usage of Fuzz4All remains the same. You can refer to their readme. 

We added:
- grazie models in Fuzz4All/models/grazie.py.
- kotlin run cofigurations in config/kotlin
- kotlin documentation to config/documentation/kotlin.
- 


