#!/bin/bash
g++ kalmanSamples.cpp `pkg-config --cflags --libs opencv`
./a.out
