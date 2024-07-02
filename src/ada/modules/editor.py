"""
editor.py - Text Editor Integration Module for ADA AI Assistant

This module provides functionality to integrate with the system's default text editor
(specifically TextEdit on macOS) for interactive editing of text content within the
ADA AI Assistant workflow.

The primary function, `edit()`, allows for opening a temporary file with given content
in TextEdit, waiting for the user to make edits, and then returning the modified content.

Key Features:
1. Creates a temporary file with a random name to avoid conflicts.
2. Opens the default text editor (TextEdit) with the temporary file.
3. Waits for the user to close the editor before proceeding.
4. Reads and returns the modified content from the temporary file.
5. Cleans up by removing the temporary file after use.

Usage:
    from modules import editor

    modified_content = editor.edit("Initial content to edit")
    print(modified_content)

Note:
- This module is designed specifically for macOS and uses the 'open' command to launch TextEdit.
- The temporary file is created in the current working directory with read-write permissions for all users.
- A 1-second delay is introduced before opening the file to ensure it's fully written to disk.

Dependencies:
- subprocess: For running system commands to open TextEdit.
- os: For file and directory operations.
- random: For generating random numbers for unique filenames.
- time: For introducing a delay before opening the editor.

This module is part of the ADA AI Assistant project and is typically used in workflows
that require human intervention or editing of AI-generated content.
"""

import os
import random
import subprocess
import time


def edit(contents: str):
    """
    Opens TextEdit on macOS and waits until it is closed to proceed.
    """
    # Get the current working directory
    current_dir = os.getcwd()
    # Generate a random number to include in the filename
    random_number = random.randint(1000, 9999)
    temp_file_path = os.path.join(current_dir, f"tempfile_{random_number}.json")

    # Create and close the temporary file explicitly
    with open(temp_file_path, "w+") as tmp:
        tmp.write(contents)
        tmp.flush()

    # Change the file permissions to make it readable and writable by everyone
    os.chmod(temp_file_path, 0o666)

    # Introduce a delay
    time.sleep(1)  # Wait for 1 second before opening the file in Editor

    # Open the default text editor and wait for it to close
    editor_process = subprocess.Popen(
        ["open", "-W", "-n", "-a", "TextEdit", temp_file_path]
    )

    # Wait for the TextEdit process to close
    editor_process.wait()

    # Read the modified content from the file
    with open(temp_file_path, "r") as file:
        modified_content = file.read()

    # Clean up by removing the temporary file
    os.remove(temp_file_path)

    return modified_content


# Example usage:
if __name__ == "__main__":
    sample_contents = "How are you doing this. Tell me more about it:"
    modified_config = edit(sample_contents)
    print(modified_config)
