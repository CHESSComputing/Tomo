#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import sys
import yaml
import pyinputplus as pyip
import matplotlib.pyplot as plt
import logging

def readConfigFile(config_filepath):
    with open(config_filepath, 'r') as f:
        lines = f.read().splitlines()
    config = {}
    for line in lines:
        line = line.split('#')[0]
        if '=' in line:
            key, value = tuple(line.split('='))
            key = key.replace(' ', '')
            config[key] = eval(value)
    return config

def searchConfigFile(config_filepath, search_string):
    with open(config_filepath, 'r') as f:
        lines = f.read()
        if search_string in lines: return True
    return False

def appendConfigFile(config_filepath, newlines):
    with open(config_filepath, 'a') as f:
        f.write('\n')
        for line in newlines.splitlines(): f.write(line + '\n')
    # update config in memory
    return readConfigFile(config_filepath)

def updateConfigFile(config_filepath, keyword, keyvalue):
    with open(config_filepath, 'r') as f:
        lines = f.read().splitlines()
    update = False
    for index in range(len(lines)):
        line = lines[index].split('#')[0]
        if keyword in line:
            key, value = tuple(line.split('='))
            key = key.replace(' ', '')
            lines[index] = key + ' = ' + str(keyvalue)
            update = True
            break
    if not update: lines += ['', keyword + ' = ' + str(keyvalue)]
    with open(config_filepath, 'w') as f: 
        for line in lines: f.write(line + '\n')
    # update config in memory
    return readConfigFile(config_filepath)

def addtoConfigFile(config_filepath, search_string, newlines):
    with open(config_filepath, 'r') as f:
        lines = f.read().splitlines()
    update = False
    for index in range(len(lines)):
        line = lines[index]
        if search_string in line:
            lines = lines[:index+1] + newlines.splitlines() + lines[index+1:]
            update = True
            break
    if not update: lines += [''] + newlines.splitlines()
    with open(config_filepath, 'w') as f:
        for line in lines: f.write(line + '\n')
    # update config in memory
    return readConfigFile(config_filepath)

def readDetectorConfig(config_filepath):
    detector = {}
    with open(config_filepath, 'r') as f:
        detector = yaml.safe_load(f)
    return detector

def get_num_files(files, file_type, num_angles = None):
    num_files = len(files)
    # RV assume that the order is correct and that the angles match the images
    if num_angles is not None and num_files >= num_angles: return num_angles
    if num_files:
        if num_files == 1:
            logging.debug('Found ' + str(num_files) + ' ' + file_type)
        else:
            logging.debug('Found ' + str(num_files) + ' ' + file_type + 's')
            num_files = pyip.inputInt('How many would you like to use (enter 0 for all)?: ', 
                    min=0, max=num_files)
            if not num_files: num_files = len(files)
    else:
        sys.exit('Unable to find any ' + file_type)
    return num_files

# use pyinputplus instead of using this (python3)
"""
def get_int_in_range(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
        except NameError or ValueError:
            print('Illegal input, try again')
            continue
        if (min_value and value < min_value) or (max_value and value > max_value):
            print('Input out of range, try again')
            continue
        else:
            break
    return value
"""

# use pyinputplus instead of using this (python3)
"""
def get_yes_no(prompt):
    while True:
        try:
            value = input(prompt).lower().strip()[0]
        except NameError or ValueError:
            print('Illegal input, try again')
            continue
        if value != 'y' and value != 'n':
            print('Illegal input, try again')
            continue
        else:
            break
    return value == 'y'
"""

def quick_imshow(a, title=None, save_figname=None, clear=False, **kwargs):
    if clear:
        if title: plt.close(fig=title)
        else: plt.clf()
    plt.ion()
    plt.figure(title)
    plt.imshow(a, **kwargs)
    if save_figname: plt.savefig(save_figname)
    plt.pause(1)

def quick_plot(y, title=None, clear=False):
    if clear:
        if title: plt.close(fig=title)
        else: plt.clf()
    plt.ion()
    plt.figure(title)
    plt.plot(y)
    plt.pause(1)

def quick_xyplot(x, y, title=None, clear=False):
    if clear:
        if title: plt.close(fig=title)
        else: plt.clf()
    plt.ion()
    plt.figure(title)
    plt.plot(x, y)
    plt.pause(1)
