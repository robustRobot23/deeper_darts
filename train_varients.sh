#!/bin/bash

# List of bash commands
commands=(
    "echo 'First command'"
    "ls -l"
    "echo 'Another command'"
    # Add more commands as needed
)

# Execute each command in the list
for cmd in "${commands[@]}"; do
    eval "$cmd"  # Execute the command
    python predictv8.py  # Run the Python script
done
