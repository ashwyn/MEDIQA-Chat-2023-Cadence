#!/bin/bash
rm -rf outputs/temp/taskC-summarizer/*
conda run --no-capture-output -n Cadence_tasks_venv python3 ./decode_taskC.py "$@"