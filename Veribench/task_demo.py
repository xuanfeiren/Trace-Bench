#!/usr/bin/env python3
"""
Demo script to present the first task in the Veribench dataset.
Dataset: https://huggingface.co/datasets/allenanie/veribench_with_prompts
"""

import json
from datasets import load_dataset
from pprint import pprint

# ANSI color codes
class Colors:
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'        # Blue
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'         # Bold
    UNDERLINE = '\033[4m'    # Underline
    END = '\033[0m'          # Reset


def print_separator(title="", width=80, color=Colors.CYAN):
    """Print a formatted separator with optional title."""
    if title:
        title = f" {title} "
        padding = (width - len(title)) // 2
        print(f"{color}{Colors.BOLD}{'=' * padding}{title}{'=' * (width - len(title) - padding)}{Colors.END}")
    else:
        print(f"{color}{'=' * width}{Colors.END}")


def format_value(key, value, indent=0):
    """Format a value for pretty printing with colors - NO TRUNCATION."""
    spaces = "  " * indent
    
    if isinstance(value, str):
        lines = value.split('\n')
        if len(lines) > 1:  # Multi-line string
            print(f"{spaces}{Colors.YELLOW}{Colors.BOLD}{key}:{Colors.END}")
            # Show ALL lines - no truncation
            for i, line in enumerate(lines):
                print(f"{spaces}  {Colors.BLUE}{i+1:3d}:{Colors.END} {line}")
        else:
            # Show complete single line - no truncation
            print(f"{spaces}{Colors.YELLOW}{Colors.BOLD}{key}:{Colors.END} {value}")
    elif isinstance(value, (list, tuple)):
        print(f"{spaces}{Colors.YELLOW}{Colors.BOLD}{key}:{Colors.END} {Colors.GREEN}[{len(value)} items]{Colors.END}")
        # Show ALL items - no truncation
        for i, item in enumerate(value):
            if isinstance(item, dict):
                print(f"{spaces}  {Colors.CYAN}[{i}]:{Colors.END} {Colors.GREEN}{{dict with {len(item)} keys}}{Colors.END}")
                # Show ALL keys - no truncation
                for k, v in item.items():
                    format_value(k, v, indent + 2)
            else:
                format_value(f"[{i}]", item, indent + 1)
    elif isinstance(value, dict):
        print(f"{spaces}{Colors.YELLOW}{Colors.BOLD}{key}:{Colors.END} {Colors.GREEN}{{dict with {len(value)} keys}}{Colors.END}")
        # Show ALL keys - no truncation
        for k, v in value.items():
            format_value(k, v, indent + 1)
    else:
        print(f"{spaces}{Colors.YELLOW}{Colors.BOLD}{key}:{Colors.END} {Colors.GREEN}{value}{Colors.END}")


def main():
    """Load and display the first task from the Veribench dataset."""
    
    print_separator("VERIBENCH DATASET DEMO", color=Colors.HEADER)
    print(f"{Colors.CYAN}Loading Veribench dataset from Hugging Face...{Colors.END}")
    print(f"{Colors.BLUE}Dataset: allenanie/veribench_with_prompts{Colors.END}")
    print()
    
    try:
        # Load the dataset
        dataset = load_dataset("allenanie/veribench_with_prompts")
        
        print(f"{Colors.GREEN}✓ Dataset loaded successfully!{Colors.END}")
        print(f"{Colors.BLUE}Available splits: {Colors.BOLD}{list(dataset.keys())}{Colors.END}")
        
        # Get the first split (usually 'train')
        split_name = list(dataset.keys())[0]
        split_data = dataset[split_name]
        
        print(f"{Colors.BLUE}Using split: {Colors.BOLD}'{split_name}'{Colors.END}")
        print(f"{Colors.BLUE}Number of examples in {split_name}: {Colors.BOLD}{len(split_data)}{Colors.END}")
        print()
        
        if len(split_data) == 0:
            print(f"{Colors.RED}No examples found in the dataset!{Colors.END}")
            return
            
        # Get the first task
        first_task = split_data[0]
        
        print_separator("FIRST TASK DETAILS", color=Colors.GREEN)
        print(f"{Colors.BLUE}Task type: {Colors.BOLD}{type(first_task)}{Colors.END}")
        print(f"{Colors.BLUE}Number of fields: {Colors.BOLD}{len(first_task) if isinstance(first_task, dict) else 'N/A'}{Colors.END}")
        print()
        
        # Display all fields in the first task - COMPLETE CONTENT
        if isinstance(first_task, dict):
            print(f"{Colors.HEADER}{Colors.BOLD}All fields in the first task (COMPLETE - NO TRUNCATION):{Colors.END}")
            print()
            
            # Print ALL content completely - no limits, no truncation
            for key, value in first_task.items():
                format_value(key, value)
                print()
        else:
            print(f"{Colors.HEADER}First task content:{Colors.END}")
            pprint(first_task)
            
        print_separator("END OF DEMO", color=Colors.HEADER)
        
    except Exception as e:
        print(f"{Colors.RED}Error loading dataset: {e}{Colors.END}")
        print(f"{Colors.YELLOW}Make sure you have:{Colors.END}")
        print(f"{Colors.BLUE}1. Internet connection{Colors.END}")
        print(f"{Colors.BLUE}2. datasets library installed{Colors.END}")
        print(f"{Colors.BLUE}3. Hugging Face authentication if required{Colors.END}")


if __name__ == "__main__":
    main()