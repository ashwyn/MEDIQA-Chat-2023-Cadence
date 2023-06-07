#!/bin/bash
rm -rf outputs/temp/taskB-summarizer/*
conda run --no-capture-output -n Cadence_tasks_venv python3 ./decode_taskB_run2.py "$@"