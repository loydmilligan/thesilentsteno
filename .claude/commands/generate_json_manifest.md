Claude Code Manifest Generator Promptclaude-code "You are a code analysis agent. Your ONLY task is to analyze the source code of a project and generate a detailed JSON manifest. You MUST NOT summarize the project. You MUST follow the JSON schema provided.

## Task: Generate codebase_manifest.json

Start your response with: "🔍 **GENERATE_MANIFEST EXECUTING** - Performing detailed code analysis and creating codebase_manifest.json"

## CRITICAL INSTRUCTIONS:
1.  **OUTPUT FORMAT IS JSON:** The final output MUST be a single, valid JSON file named `codebase_manifest.json`. Do NOT output Markdown or any other format.
2.  **PERFORM DEEP CODE ANALYSIS:** You MUST scan every source file (`.js`, `.jsx`, `.ts`, `.tsx`, `.py`, etc.) individually. Do NOT rely on `package.json` or `README.md` for the file analysis section.
3.  **DO NOT SUMMARIZE:** Your task is to create a structured representation of the code, not a human-readable summary. The `files` object in the JSON must contain an entry for every single source file.
4.  **ADHERE TO THE SCHEMA:** The generated JSON must strictly follow the schema defined below. Every required field must be present.

## Manifest JSON Schema:

```json
{
  "version": "1.0",
  "generated": "[current timestamp in ISO format]",
  "project": "[infer project name from package.json or directory]",
  "description": "[brief description from README or package.json]",
  "files": {
    "path/to/file.ext": {
      "purpose": "[one-line, inferred description of this file's specific role]",
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
      "imports": ["list of all imported modules and local files"],
      "sideEffects": ["list of side effects like 'writes-database', 'network-calls', etc."]
    }
  },
  "dependencies": {
    "[package-name]": "[brief description of what this dependency provides]"
  },
  "architecture": {
    "main_flow": "[describe the main execution flow, inferred from code]",
    "data_flow": "[describe how data flows through the system, inferred from code]",
    "configuration": "[describe how the system is configured]"
  }
}
Analysis Instructions:Scan all source code files in the current directory and subdirectories.Ignore these files/directories:node_modules/.git/dist/build/.DS_Store*.log files.env files (but note if they exist in the configuration section)For each file, you MUST determine:Purpose: A concise, one-sentence description of the file's role, inferred from its contents.Exports: A detailed list of all functions, classes, and constants the file exports. This is mandatory.Imports: A complete list of all external packages and local files it imports.Side Effects: What the file does beyond pure computation, based on the categories below.For exports, provide complete API documentation as per the schema. This includes full function signatures, parameters, return values, and descriptions.Side Effects Categories:'writes-database' - modifies persistent storage'reads-database' - reads from persistent storage'network-calls' - makes HTTP/API calls'sends-data' - sends data over a persistent connection (e.g., WebSocket, TCP)'receives-data' - receives data over a persistent connection'publishes-events' - sends messages to a pub/sub system'subscribes-to-events' - listens for messages from a pub/sub system'writes-files' - creates or modifies files'reads-files' - reads from files'creates-ui' - creates user interface elements'modifies-dom' - changes DOM elements'registers-events' - sets up event listeners'registers-commands' - adds commands to systems'loads-settings' - reads configuration'saves-settings' - writes configurationFor package.json dependencies, read the actual dependencies and provide brief descriptions for the dependencies key in the JSON.Architecture Analysis: Infer the architecture from the code itself, not just from documentation. Trace the primary execution flow from the entry point.Final Output Requirement:Create the file named codebase_manifest.json.The file's content must be only the JSON object described
