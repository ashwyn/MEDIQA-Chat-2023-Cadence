#!/bin/bash
rm -rf outputs/temp/taskA-summarizer/*
conda run --no-capture-output -n Cadence_tasks_venv python3 ./decode_taskA.py "$@"