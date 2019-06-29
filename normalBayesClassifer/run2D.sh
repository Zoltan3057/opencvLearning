#!/bin/bash
g++ -g -o demo2D demo2D.cpp `pkg-config --cflags --libs opencv`
./demo2D
