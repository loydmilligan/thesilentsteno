### **Claude Code Manifest Generator Prompt**

claude-code "Please analyze the current codebase and generate a comprehensive manifest file following these specifications:

## Task: Generate codebase_manifest.json

Create a manifest file that maps the current state of the codebase. Scan all files and analyze their structure to create an accurate representation.

Start your response with: "🔍 **GENERATE_MANIFEST EXECUTING** - Analyzing current codebase and creating manifest"

## Manifest Format Required:

```json
{
  "version": "1.0",
  "generated": "[current timestamp in ISO format]",
  "project": "[infer project name from package.json or directory]",
  "description": "[brief description of what this codebase does]",
  "files": {
    "path/to/file.ext": {
      "purpose": "[one line description of what this file does]",
      "exports": {
        "functions": [
          {
            "name": "function_name",
            "signature": "function_name(param1: type, param2: type) -> return_type",
            "description": "what this function does",
            "parameters": {
              "param1": "description of param1",
              "param2": "description of param2"
            },
            "returns": "description of return value"
          }
        ],
        "classes": [
          {
            "name": "ClassName",
            "description": "what this class does",
            "constructor": "ClassName(param1: type, param2: type)",
            "methods": [
              {
                "name": "method_name",
                "signature": "method_name(param: type) -> return_type",
                "description": "what this method does",
                "parameters": {"param": "description"},
                "returns": "description of return value"
              }
            ],
            "properties": [
              {
                "name": "property_name",
                "type": "property_type",
                "description": "what this property stores"
              }
            ]
          }
        ],
        "constants": [
          {
            "name": "CONSTANT_NAME",
            "type": "constant_type",
            "value": "actual_value_or_description",
            "description": "what this constant represents"
          }
        ]
      },
      "imports": ["list of main dependencies and local imports"],
      "sideEffects": ["list of side effects like 'writes-database', 'network-calls', 'modifies-files', 'creates-ui', etc."]
    }
  },
  "dependencies": {
    "[package-name]": "[brief description of what this dependency provides]"
  },
  "architecture": {
    "main_flow": "[describe the main execution flow]",
    "data_flow": "[describe how data flows through the system]",
    "configuration": "[describe how the system is configured]"
  }
}
```

## **Analysis Instructions:**

1. **Scan all files** in the current directory and subdirectories
2. **Ignore** these files/directories:
   * node_modules/
   * .git/
   * dist/
   * build/
   * .DS_Store
   * *.log files
   * .env files (but note if they exist)
3. **For each file, determine:**
   * **Purpose**: What does this file do? (one concise sentence)
   * **Exports**: What functions, classes, constants, or types does it export? Include full API details.
   * **Imports**: What external packages and local files does it import?
   * **Side Effects**: What does it do beyond pure computation?
4. **For exports, provide complete API documentation:**
   * **Functions**: Include signature, parameters, return values, and description
   * **Classes**: Include constructor, all public methods, properties, and descriptions
   * **Constants**: Include type, value, and purpose
   * **Method details**: Include parameter types, return types, and what each method does
5. **Side Effects Categories:**
   * 'writes-database' - modifies persistent storage
   * 'reads-database' - reads from persistent storage
   * 'network-calls' - makes HTTP/API calls
   * 'sends-data' - sends data over a persistent connection (e.g., WebSocket, TCP)
   * 'receives-data' - receives data over a persistent connection
   * 'publishes-events' - sends messages to a pub/sub system (e.g., message queue, event bus)
   * 'subscribes-to-events' - listens for messages from a pub/sub system
   * 'writes-files' - creates or modifies files
   * 'reads-files' - reads from files
   * 'creates-ui' - creates user interface elements
   * 'modifies-dom' - changes DOM elements
   * 'registers-events' - sets up event listeners
   * 'registers-commands' - adds commands to systems
   * 'loads-settings' - reads configuration
   * 'saves-settings' - writes configuration
6. **For package.json dependencies**, read the actual dependencies and provide brief descriptions of what each major dependency does.
7. **Architecture Analysis:**
   * Identify the main entry point
   * Trace the primary execution flow
   * Identify how data moves through the system
   * Note how configuration is handled

## **Output Requirements:**

* Create the file as codebase_manifest.json in the root directory
* Use proper JSON formatting with proper escaping
* Include all files that contain actual code (not just config files)
* Be accurate about exports - read the actual export statements
* Be accurate about imports - read the actual import statements
* If a file doesn't exist yet but is referenced, note it in the manifest with "status": "planned"

## **Example Analysis Process:**

If this is the directory tree of the current codebase:

```
file-sorter/
├── main.py
├── file_sorter.py
├── config.py
├── README.md
├── .gitignore
└── requirements.txt
```

With these file contents:

**main.py:**
```python
#!/usr/bin/env python3
import sys
from file_sorter import FileSorter
from config import Config

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <directory>")
        sys.exit(1)
    
    config = Config()
    sorter = FileSorter(config)
    sorter.sort_directory(sys.argv[1])

if __name__ == "__main__":
    main()
```

**file_sorter.py:**
```python
import os
import shutil
from pathlib import Path

class FileSorter:
    def __init__(self, config):
        self.config = config
    
    def sort_directory(self, directory):
        """Sort files in directory by type"""
        for filename in os.listdir(directory):
            file_path = Path(directory) / filename
            if file_path.is_file():
                self._move_file(file_path)
    
    def _move_file(self, file_path):
        """Move file to appropriate subdirectory"""
        extension = file_path.suffix.lower()
        target_dir = self.config.get_target_directory(extension)
        
        target_path = file_path.parent / target_dir
        target_path.mkdir(exist_ok=True)
        
        shutil.move(str(file_path), str(target_path / file_path.name))
```

**config.py:**
```python
class Config:
    def __init__(self):
        self.file_mappings = {
            '.txt': 'documents',
            '.pdf': 'documents', 
            '.jpg': 'images',
            '.png': 'images',
            '.mp4': 'videos',
            '.mp3': 'audio'
        }
    
    def get_target_directory(self, extension):
        """Get target directory for file extension"""
        return self.file_mappings.get(extension, 'misc')
```

**README.md:** (non-code file - should be skipped)
```markdown
# File Sorter
A simple Python script to sort files by type into subdirectories.
```

**.gitignore:** (non-code file - should be skipped)
```
__pycache__/
*.pyc
.DS_Store
```

The analysis would produce:

1. **Read main.py** - imports sys, file_sorter, config; exports main(); side effects: reads-files, creates-directories
2. **Read file_sorter.py** - imports os, shutil, pathlib; exports FileSorter class; side effects: writes-files, creates-directories
3. **Read config.py** - no imports; exports Config class; no side effects
4. **Skip README.md** - documentation file
5. **Skip .gitignore** - configuration file

**Expected Generated Manifest:**

```json
{
  "version": "1.0",
  "generated": "2025-07-03T15:30:00Z",
  "project": "file-sorter",
  "description": "A Python script that sorts files by type into subdirectories",
  "files": {
    "main.py": {
      "purpose": "Main entry point that handles command line arguments and orchestrates file sorting",
      "exports": {
        "functions": [
          {
            "name": "main",
            "signature": "main() -> None",
            "description": "Main entry point that processes command line arguments and runs file sorting",
            "parameters": {},
            "returns": "None"
          }
        ],
        "classes": [],
        "constants": []
      },
      "imports": ["sys", "file_sorter.FileSorter", "config.Config"],
      "sideEffects": ["reads-files", "creates-directories"]
    },
    "file_sorter.py": {
      "purpose": "Contains FileSorter class that handles moving files to appropriate directories",
      "exports": {
        "functions": [],
        "classes": [
          {
            "name": "FileSorter",
            "description": "Sorts files in a directory by file type into subdirectories",
            "constructor": "FileSorter(config: Config)",
            "methods": [
              {
                "name": "sort_directory",
                "signature": "sort_directory(directory: str) -> None",
                "description": "Sort all files in the given directory by type into subdirectories",
                "parameters": {
                  "directory": "Path to directory containing files to sort"
                },
                "returns": "None"
              },
              {
                "name": "_move_file",
                "signature": "_move_file(file_path: Path) -> None",
                "description": "Move a single file to appropriate subdirectory based on extension",
                "parameters": {
                  "file_path": "Path object representing the file to move"
                },
                "returns": "None"
              }
            ],
            "properties": [
              {
                "name": "config",
                "type": "Config",
                "description": "Configuration object containing file type mappings"
              }
            ]
          }
        ],
        "constants": []
      },
      "imports": ["os", "shutil", "pathlib.Path"],
      "sideEffects": ["writes-files", "creates-directories"]
    },
    "config.py": {
      "purpose": "Configuration class that defines file type to directory mappings",
      "exports": {
        "functions": [],
        "classes": [
          {
            "name": "Config",
            "description": "Handles configuration for file type to directory mappings",
            "constructor": "Config()",
            "methods": [
              {
                "name": "get_target_directory",
                "signature": "get_target_directory(extension: str) -> str",
                "description": "Get the target directory name for a given file extension",
                "parameters": {
                  "extension": "File extension (e.g., '.txt', '.jpg')"
                },
                "returns": "Directory name where files of this type should be stored"
              }
            ],
            "properties": [
              {
                "name": "file_mappings",
                "type": "dict[str, str]",
                "description": "Dictionary mapping file extensions to target directory names"
              }
            ]
          }
        ],
        "constants": []
      },
      "imports": [],
      "sideEffects": []
    }
  },
  "dependencies": {},
  "architecture": {
    "main_flow": "main.py -> FileSorter.sort_directory() -> Config.get_target_directory() -> file operations",
    "data_flow": "Command line args -> Config mappings -> File system operations",
    "configuration": "File type mappings defined in Config class constructor"
  }
}
```

## **Language-Specific Considerations:**

* **JavaScript/TypeScript**: Look for exports, imports, default exports, type definitions
* **Python**: Look for classes, functions, imports, __all__ declarations
* **Java**: Look for classes, interfaces, packages, public methods
* **C#**: Look for classes, interfaces, namespaces, public members
* **Go**: Look for packages, exported functions/types (capitalized names)
* **Rust**: Look for pub items, modules, traits, structs
* **Ruby**: Look for classes, modules, methods, requires
* **PHP**: Look for classes, functions, namespaces, includes

Adapt the manifest structure to best represent the specific language and framework patterns found in the codebase."