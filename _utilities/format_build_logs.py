#!/usr/bin/env python3
"""
Format Sphinx Build Logs

This script checks for Python 3.9 and pip, creates a virtual environment,
runs sphinx-build, and formats the build log as Markdown with separate
sections for errors and warnings.
"""

import os
import sys
import subprocess
import re
import datetime
import platform
import shutil
from collections import Counter
from pathlib import Path

def check_python_version():
    """Check if Python 3.9 is installed."""
    python_version = sys.version_info
    
    if python_version.major != 3 or python_version.minor != 9:
        print("Error: Python 3.9 is required.")
        
        if platform.system() == "Darwin":  # macOS
            print("To install Python 3.9 on macOS, visit: https://www.python.org/downloads/release/python-3913/")
            print("Or use Homebrew: brew install python@3.9")
        elif platform.system() == "Windows":
            print("To install Python 3.9 on Windows, visit: https://www.python.org/downloads/release/python-3913/")
        else:
            print("Please install Python 3.9 from: https://www.python.org/downloads/release/python-3913/")
            
        sys.exit(1)
    
    return True

def check_pip_installed():
    """Check if pip is installed."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        print("Error: pip is not installed.")
        print("Please install pip: https://pip.pypa.io/en/stable/installation/")
        sys.exit(1)

def find_repo_root():
    """Find the root of the private-aws-neuron-sdk-staging repo."""
    # Start with the current directory
    current_dir = Path.cwd()
    
    # Check if we're already in the repo root
    if current_dir.name == "private-aws-neuron-sdk-staging":
        return current_dir
    
    # Check parent directory
    parent_dir = current_dir.parent
    if parent_dir.name == "private-aws-neuron-sdk-staging":
        return parent_dir
    
    # Look for the repo in the current directory
    for item in current_dir.iterdir():
        if item.is_dir() and item.name == "private-aws-neuron-sdk-staging":
            return item
    
    # Look for the repo in the parent directory
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name == "private-aws-neuron-sdk-staging":
            return item
    
    print("Error: Repository 'private-aws-neuron-sdk-staging' not found on local machine.")
    sys.exit(1)

def setup_venv(repo_parent):
    """Create and activate a Python 3.9 virtual environment."""
    venv_path = repo_parent / "venv"
    
    # Create venv if it doesn't exist
    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    
    # Determine the path to the activate script
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        activate_cmd = str(activate_script)
    else:
        activate_script = venv_path / "bin" / "activate"
        activate_cmd = f"source {activate_script}"
    
    print(f"Virtual environment created at {venv_path}")
    print(f"To activate manually, run: {activate_cmd}")
    
    return venv_path

def get_venv_python(venv_path):
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def get_venv_pip(venv_path):
    """Get the path to the pip executable in the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def install_requirements(repo_root, venv_pip):
    """Install requirements from requirements.txt."""
    requirements_file = repo_root / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        sys.exit(1)
    
    print("Installing requirements...")
    try:
        subprocess.run([
            str(venv_pip), "install", "-r", str(requirements_file),
            "--extra-index-url=https://pypi.org/simple"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)
    
    print("Requirements installed successfully.")

def run_sphinx_build(repo_root, venv_path):
    """Run sphinx-build and capture the output."""
    sphinx_build_path = venv_path / "bin" / "sphinx-build"
    if platform.system() == "Windows":
        sphinx_build_path = venv_path / "Scripts" / "sphinx-build.exe"
    
    if not sphinx_build_path.exists():
        print(f"Error: sphinx-build not found at {sphinx_build_path}")
        sys.exit(1)
    
    print("Running sphinx-build...")
    
    # Create a log file to capture output
    log_file_path = repo_root / "sphinx_build_output.log"
    
    try:
        # Run sphinx-build with output redirected to both terminal and log file
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                [str(sphinx_build_path), "-b", "html", ".", "_build/html", "-w", "warnings.txt"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Capture output in real-time
            output = []
            for line in process.stdout:
                print(line, end='')  # Print to terminal
                log_file.write(line)  # Write to log file
                output.append(line)
                
            process.wait()
            
            if process.returncode != 0:
                print(f"sphinx-build exited with code {process.returncode}")
        
        # Also read the warnings.txt file if it exists
        warnings_file = repo_root / "warnings.txt"
        if warnings_file.exists():
            with open(warnings_file, 'r') as f:
                warnings_content = f.read()
                output.append("\n--- WARNINGS FILE CONTENT ---\n")
                output.append(warnings_content)
        
        return ''.join(output)
    except Exception as e:
        print(f"Error running sphinx-build: {e}")
        sys.exit(1)

def parse_build_log(log_text):
    """Parse the build log to extract errors and warnings."""
    # Save raw log for debugging
    with open("raw_build_log.txt", "w") as f:
        f.write(log_text)
    
    # Check if warnings.txt exists and use it directly
    warnings_file = Path("warnings.txt")
    if warnings_file.exists():
        print(f"Found warnings.txt file with direct warnings from Sphinx")
        with open(warnings_file, 'r') as f:
            warnings_content = f.read()
            
        # Parse warnings.txt which has format: path:line: WARNING: message
        warnings = []
        for line in warnings_content.split('\n'):
            if not line.strip():
                continue
                
            # Try to match the standard format first
            match = re.match(r'(.*?):(\d+): WARNING: (.*)', line)
            if match:
                file_path, line_num, message = match.groups()
                warnings.append({
                    'file': file_path,
                    'line': line_num,
                    'message': message.strip()
                })
                print(f"Standard format match: file={file_path}, line={line_num}, message={message[:50]}...")
            else:
                # Check for the "document isn't included in any toctree" pattern
                # Format: /path/to/file.rst: WARNING: document isn't included in any toctree
                toctree_match = re.match(r'(.*?): WARNING: (document isn\'t included in any toctree.*)', line)
                if toctree_match:
                    file_path, message = toctree_match.groups()
                    warnings.append({
                        'file': file_path,
                        'line': '0',  # No line number in this format
                        'message': message.strip()
                    })
                    print(f"Toctree match: file={file_path}, message={message[:50]}...")
                else:
                    # If no match, just add as unknown
                    warnings.append({
                        'file': 'unknown',
                        'line': '0',
                        'message': line.strip()
                    })
                    print(f"No match: message={line[:50]}...")
    else:
        print("No warnings.txt file found, parsing log output directly")
        warnings = []
        lines = log_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
                
            # Check for the "document isn't included in any toctree" pattern
            # Format: /path/to/file.rst: WARNING: document isn't included in any toctree
            toctree_match = re.match(r'(.*?): WARNING: (document isn\'t included in any toctree.*)', line)
            if toctree_match:
                file_path, message = toctree_match.groups()
                warnings.append({
                    'file': file_path,
                    'line': '0',  # No line number in this format
                    'message': message.strip()
                })
                i += 1
                continue
                
            # Check for warnings in the raw message
            # This is for warnings that are already in the log as complete messages
            raw_warning_match = re.match(r'(.*?): WARNING: (.*)', line)
            if raw_warning_match:
                file_path, message = raw_warning_match.groups()
                warnings.append({
                    'file': file_path,
                    'line': '0',  # No line number in this format
                    'message': message.strip()
                })
                i += 1
                continue
            
            # Check for standard format: path:line: WARNING: message
            std_match = re.match(r'(.*?):(\d+): WARNING: (.*)', line)
            if std_match:
                file_path, line_num, message = std_match.groups()
                warnings.append({
                    'file': file_path,
                    'line': line_num,
                    'message': message.strip()
                })
                i += 1
                continue
                
            # Check for alternative format: WARNING: message (path:line)
            alt_match = re.match(r'WARNING: (.*?) \((.*?):(\d+)\)', line)
            if alt_match:
                message, file_path, line_num = alt_match.groups()
                warnings.append({
                    'file': file_path,
                    'line': line_num,
                    'message': message.strip()
                })
                i += 1
                continue
                
            # Check for simple warnings that start with "WARNING:"
            if line.startswith("WARNING:"):
                message = line[8:].strip()  # Remove "WARNING: " prefix
                
                # Collect continuation lines
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("WARNING:", "ERROR:")):
                    message += " " + lines[i].strip()
                    i += 1
                    
                warnings.append({
                    'file': 'unknown',
                    'line': '0',
                    'message': message
                })
                continue
                
            i += 1
    
    # Debug: Print the first few warnings to see what's being parsed
    print(f"Parsed {len(warnings)} warnings")
    for i, warning in enumerate(warnings[:5]):
        print(f"Warning {i+1}: file={warning['file']}, line={warning['line']}, message={warning['message'][:50]}...")
    
    # Debug: Print the warning categories
    categories = categorize_issues(warnings)
    print(f"Warning categories: {categories}")
    
    # Regular expressions for errors
    error_pattern = re.compile(r'(.*?):(\d+): (?:ERROR|SEVERE): (.*?)(?:\n|$)')
    
    errors = []
    lines = log_text.split('\n')
    for line in lines:
        error_match = error_pattern.search(line)
        if error_match:
            file_path, line_num, message = error_match.groups()
            errors.append({
                'file': file_path,
                'line': line_num,
                'message': message.strip()
            })
    
    return errors, warnings

def categorize_issues(issues):
    """Categorize issues by type."""
    categories = Counter()
    
    for issue in issues:
        # Extract the main category from the message
        message = issue['message'].lower()
        
        if "undefined label" in message:
            categories["Undefined Label"] += 1
        elif "unknown document" in message:
            categories["Unknown Document"] += 1
        elif "duplicate label" in message:
            categories["Duplicate Label"] += 1
        elif "image file not found" in message:
            categories["Missing Image"] += 1
        elif "toctree contains reference to nonexisting document" in message:
            categories["Missing Document"] += 1
        elif "document isn't included in any toctree" in message:
            categories["Document Not in TOC"] += 1
        else:
            categories["Other"] += 1
    
    return categories

def format_markdown(errors, warnings, build_time):
    """Format the build log as Markdown."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    error_categories = categorize_issues(errors)
    warning_categories = categorize_issues(warnings)
    
    markdown = f"# Sphinx Build Log - {timestamp}\n\n"
    
    # Build summary
    markdown += "## Build Summary\n\n"
    markdown += f"- **Build Time**: {build_time:.2f} seconds\n"
    markdown += f"- **Total Errors**: {len(errors)}\n"
    markdown += f"- **Total Warnings**: {len(warnings)}\n\n"
    
    # Error categories
    if error_categories:
        markdown += "### Error Categories\n\n"
        for category, count in error_categories.most_common():
            markdown += f"- **{category}**: {count}\n"
        markdown += "\n"
    
    # Warning categories
    if warning_categories:
        markdown += "### Warning Categories\n\n"
        for category, count in warning_categories.most_common():
            markdown += f"- **{category}**: {count}\n"
        markdown += "\n"
    
    # Errors section
    markdown += "## Errors\n\n"
    if errors:
        for i, error in enumerate(errors, 1):
            # Format the file path to be more readable
            file_path = error['file']
            if file_path.startswith('/Users/dougeric/git/private-aws-neuron-sdk-staging/'):
                file_path = file_path[len('/Users/dougeric/git/private-aws-neuron-sdk-staging/'):]
            
            # Create a more readable header with file and line info
            if error['file'] != 'unknown':
                markdown += f"### Error {i}: {file_path} (line {error['line']})\n\n"
            else:
                markdown += f"### Error {i}\n\n"
                
            markdown += f"```\n{error['message']}\n```\n\n"
    else:
        markdown += "No errors found.\n\n"
    
    # Warnings section
    markdown += "## Warnings\n\n"
    if warnings:
        for i, warning in enumerate(warnings, 1):
            # Format the file path to be more readable
            file_path = warning['file']
            if file_path.startswith('/Users/dougeric/git/private-aws-neuron-sdk-staging/'):
                file_path = file_path[len('/Users/dougeric/git/private-aws-neuron-sdk-staging/'):]
            
            # Create a more readable header with file and line info
            if warning['file'] != 'unknown':
                if warning['line'] != '0':
                    markdown += f"### Warning {i}: {file_path} (line {warning['line']})\n\n"
                else:
                    markdown += f"### Warning {i}: {file_path}\n\n"
            else:
                markdown += f"### Warning {i}\n\n"
                
            # Don't include the file path in the message if it's already in the header
            message = warning['message']
            if warning['file'] != 'unknown' and message.startswith(warning['file']):
                # Remove the file path from the message
                message = message[len(warning['file'])+2:] # +2 for ": "
            
            markdown += f"```\n{message}\n```\n\n"
    else:
        markdown += "No warnings found.\n\n"
    
    return markdown

def main():
    """Main function."""
    print("Checking Python version...")
    check_python_version()
    
    print("Checking pip installation...")
    check_pip_installed()
    
    print("Finding repository root...")
    repo_root = find_repo_root()
    repo_parent = repo_root.parent
    
    print(f"Repository found at: {repo_root}")
    
    print("Setting up virtual environment...")
    venv_path = setup_venv(repo_parent)
    venv_python = get_venv_python(venv_path)
    venv_pip = get_venv_pip(venv_path)
    
    print(f"Changing directory to {repo_root}...")
    os.chdir(str(repo_root))
    
    print("Installing requirements...")
    install_requirements(repo_root, venv_pip)
    
    print("Running sphinx-build...")
    start_time = datetime.datetime.now()
    build_log = run_sphinx_build(repo_root, venv_path)
    end_time = datetime.datetime.now()
    build_time = (end_time - start_time).total_seconds()
    
    print("Parsing build log...")
    errors, warnings = parse_build_log(build_log)
    
    print("Formatting build log as Markdown...")
    markdown = format_markdown(errors, warnings, build_time)
    
    # Write the formatted log to a file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = repo_root / f"build-log-{timestamp}.md"
    
    with open(output_file, "w") as f:
        f.write(markdown)
    
    print(f"Build log written to {output_file}")
    print(f"Found {len(errors)} errors and {len(warnings)} warnings.")

if __name__ == "__main__":
    main()
