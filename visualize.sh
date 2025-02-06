#!/bin/bash

# Check if /output directory exists
if [ ! -d "./output" ]; then
    echo "Error: /output directory does not exist."
    exit 1
fi

# Run the visualize.py script
python3 visualize.py