#!/bin/bash
g++ -g -o demo demo.cpp `pkg-config --cflags --libs opencv`
./demo
