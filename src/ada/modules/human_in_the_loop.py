"""
human_in_the_loop.py - Human Interaction Module for ADA AI Assistant

This module provides functionality to incorporate human interaction within the
ADA AI Assistant workflow. It offers methods for opening files, editing content,
and interacting with the system's default text editor.

Key Components:
1. File Selection: Allows users to choose files through a graphical interface.
2. Text Editing: Enables opening the system's default text editor for content modification.
3. File Handling: Manages opening files in the default editor for further editing.

Main Functions:
- open_file(): Opens a file selection dialog and returns the selected file path.
- open_editor(): Opens the default text editor with empty content for user input.
- open_file_in_editor_and_continue(file): Opens a specified file in the default editor.

Usage:
    from modules import human_in_the_loop

    # Select a file
    file_path = human_in_the_loop.open_file()

    # Open editor for user input
    user_input = human_in_the_loop.open_editor()

    # Open a file in the editor
    human_in_the_loop.open_file_in_editor_and_continue(file_path)

Dependencies:
- tkinter: For creating the file selection dialog.
- subprocess: For running system commands to open the text editor.
- modules.editor: Custom module for text editor integration.

Note:
This module is designed to work seamlessly with the ADA AI Assistant, providing
a bridge between automated processes and human intervention. It's particularly
useful for tasks that require human review, modification, or input within an
otherwise automated workflow.

The module assumes that the system has a default text editor configured (TextEdit
for macOS). For other operating systems, modifications may be necessary to ensure
compatibility with the appropriate text editor.
"""

import subprocess
import tkinter as tk
from tkinter import filedialog

from ada.modules import editor


def open_file() -> str:
    """Opens a file selection dialog and returns the selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfile()
    if not file_path:
        return None
    root.destroy()
    return file_path.name


def open_editor() -> str:
    return editor.edit(contents="")


def open_file_in_editor_and_continue(file: str) -> None:
    """Opens a file in the editor using the 'code' command and allows the user to continue editing."""
    if file:
        subprocess.run(["code", file])
    else:
        print("No file provided to open.")
