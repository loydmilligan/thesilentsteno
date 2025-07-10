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
  "project": {
    "name": "[infer project name from package.json, setup.py, or directory]",
    "description": "[brief description of what this codebase does]",
    "version": "[version from package file if available]",
    "tech_stack": "[primary technologies used]",
    "deployment": "[how the project is deployed/run]",
    "repository": "[repository info if available]"
  },
  "documentation": {
    "readme": "[path to readme file if exists]",
    "architecture_notes": "[brief architecture description]"
  },
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
        ],
        "types": [
          {
            "name": "TypeName",
            "definition": "type definition",
            "description": "what this type represents"
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
    "configuration": "[describe how the system is configured]",
    "entry_points": ["list of main entry points"]
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
   * target/
   * bin/
   * obj/
   * __pycache__/
   * .DS_Store
   * *.log files
   * .env files (but note if they exist)
   * .cache/
   * .tmp/
   * vendor/ (for some languages)

3. **For each code file, determine:**
   * **Purpose**: What does this file do? (one concise sentence)
   * **Exports**: What functions, classes, constants, types, or modules does it export? Include full API details.
   * **Imports**: What external packages and local files does it import?
   * **Side Effects**: What does it do beyond pure computation?

4. **For exports, provide complete API documentation:**
   * **Functions**: Include signature, parameters, return values, and description
   * **Classes**: Include constructor, all public methods, properties, and descriptions
   * **Constants**: Include type, value, and purpose
   * **Types**: Include type definitions and purpose
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
   * 'starts-servers' - launches server processes
   * 'connects-services' - establishes connections to external services

6. **For dependencies**, read the actual dependency files (package.json, requirements.txt, go.mod, etc.) and provide brief descriptions of what each major dependency does.

7. **Architecture Analysis:**
   * Identify the main entry point(s)
   * Trace the primary execution flow
   * Identify how data moves through the system
   * Note how configuration is handled
   * Identify key architectural patterns used

## **Output Requirements:**

* Create the file as codebase_manifest.json in the root directory
* Use proper JSON formatting with proper escaping
* Include all files that contain actual code (not just config files)
* Be accurate about exports - read the actual export statements
* Be accurate about imports - read the actual import statements
* If a file doesn't exist yet but is referenced, note it in the manifest with "status": "planned"
* Adapt the analysis to the specific language(s) and frameworks being used
* For configuration files, note their purpose but don't analyze them as code files

## **Language-Specific Considerations:**

* **JavaScript/TypeScript**: Look for exports, imports, default exports, type definitions
* **Python**: Look for classes, functions, imports, __all__ declarations
* **Java**: Look for classes, interfaces, packages, public methods
* **C#**: Look for classes, interfaces, namespaces, public members
* **Go**: Look for packages, exported functions/types (capitalized names)
* **Rust**: Look for pub items, modules, traits, structs
* **Ruby**: Look for classes, modules, methods, requires
* **PHP**: Look for classes, functions, namespaces, includes

Adapt the manifest structure to best represent the specific language and framework patterns found in the codebase.

## **Example Analysis Process:**

If this is the directory tree of the current codebase:

```
todo-app/
├── package.json
├── index.js
├── src/
│   ├── models/
│   │   └── task.js
│   ├── routes/
│   │   └── api.js
│   └── utils/
│       └── validator.js
├── public/
│   ├── index.html
│   └── script.js
└── README.md
```

With these file contents:

**package.json:**
```json
{
  "name": "todo-app",
  "version": "1.0.0",
  "description": "A simple task management web application",
  "main": "index.js",
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5"
  }
}
```

**index.js:**
```javascript
const express = require('express');
const cors = require('cors');
const apiRoutes = require('./src/routes/api');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/api', apiRoutes);

function startServer() {
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
}

if (require.main === module) {
    startServer();
}

module.exports = { app, startServer };
```

**src/models/task.js:**
```javascript
class Task {
    constructor(id, title, completed = false) {
        this.id = id;
        this.title = title;
        this.completed = completed;
        this.createdAt = new Date();
    }

    toggle() {
        this.completed = !this.completed;
        return this;
    }

    update(title) {
        this.title = title;
        return this;
    }
}

const TASK_STATUS = {
    PENDING: 'pending',
    COMPLETED: 'completed'
};

module.exports = { Task, TASK_STATUS };
```

**src/routes/api.js:**
```javascript
const express = require('express');
const { Task } = require('../models/task');
const { validateTask } = require('../utils/validator');

const router = express.Router();
let tasks = [];
let nextId = 1;

router.get('/tasks', (req, res) => {
    res.json(tasks);
});

router.post('/tasks', (req, res) => {
    const validation = validateTask(req.body);
    if (!validation.isValid) {
        return res.status(400).json({ error: validation.error });
    }
    
    const task = new Task(nextId++, req.body.title);
    tasks.push(task);
    res.status(201).json(task);
});

module.exports = router;
```

**src/utils/validator.js:**
```javascript
function validateTask(taskData) {
    if (!taskData.title || typeof taskData.title !== 'string') {
        return { isValid: false, error: 'Title is required and must be a string' };
    }
    
    if (taskData.title.trim().length === 0) {
        return { isValid: false, error: 'Title cannot be empty' };
    }
    
    return { isValid: true };
}

module.exports = { validateTask };
```

**public/script.js:**
```javascript
const API_BASE = '/api';

async function fetchTasks() {
    const response = await fetch(`${API_BASE}/tasks`);
    return response.json();
}

async function createTask(title) {
    const response = await fetch(`${API_BASE}/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
    });
    return response.json();
}

function renderTasks(tasks) {
    const container = document.getElementById('tasks');
    container.innerHTML = tasks.map(task => 
        `<div class="task">${task.title}</div>`
    ).join('');
}
```

The analysis would produce:

**Expected Generated Manifest:**

```json
{
  "version": "1.0",
  "generated": "2025-07-07T15:30:00Z",
  "project": {
    "name": "todo-app",
    "description": "A simple task management web application",
    "version": "1.0.0",
    "tech_stack": "Node.js, Express.js, HTML, JavaScript",
    "deployment": "Web server on port 3000",
    "repository": "Local development"
  },
  "documentation": {
    "readme": "README.md",
    "architecture_notes": "REST API backend with static frontend, in-memory task storage"
  },
  "files": {
    "index.js": {
      "purpose": "Main server entry point that configures Express app and starts the web server",
      "exports": {
        "functions": [
          {
            "name": "startServer",
            "signature": "startServer() -> void",
            "description": "Starts the Express server on configured port",
            "parameters": {},
            "returns": "void"
          }
        ],
        "classes": [],
        "constants": [],
        "types": []
      },
      "imports": ["express", "cors", "./src/routes/api"],
      "sideEffects": ["starts-servers", "loads-settings", "creates-ui"]
    },
    "src/models/task.js": {
      "purpose": "Task model class and related constants for task management",
      "exports": {
        "functions": [],
        "classes": [
          {
            "name": "Task",
            "description": "Represents a task with id, title, completion status and timestamps",
            "constructor": "Task(id: number, title: string, completed: boolean = false)",
            "methods": [
              {
                "name": "toggle",
                "signature": "toggle() -> Task",
                "description": "Toggles the completion status of the task",
                "parameters": {},
                "returns": "The task instance for method chaining"
              },
              {
                "name": "update",
                "signature": "update(title: string) -> Task",
                "description": "Updates the task title",
                "parameters": {
                  "title": "New title for the task"
                },
                "returns": "The task instance for method chaining"
              }
            ],
            "properties": [
              {
                "name": "id",
                "type": "number",
                "description": "Unique identifier for the task"
              },
              {
                "name": "title",
                "type": "string",
                "description": "Task description text"
              },
              {
                "name": "completed",
                "type": "boolean",
                "description": "Whether the task is completed"
              },
              {
                "name": "createdAt",
                "type": "Date",
                "description": "Timestamp when task was created"
              }
            ]
          }
        ],
        "constants": [
          {
            "name": "TASK_STATUS",
            "type": "object",
            "value": "{ PENDING: 'pending', COMPLETED: 'completed' }",
            "description": "Enumeration of possible task statuses"
          }
        ],
        "types": []
      },
      "imports": [],
      "sideEffects": []
    },
    "src/routes/api.js": {
      "purpose": "Express router handling REST API endpoints for task operations",
      "exports": {
        "functions": [],
        "classes": [],
        "constants": [],
        "types": []
      },
      "imports": ["express", "../models/task", "../utils/validator"],
      "sideEffects": ["registers-events"]
    },
    "src/utils/validator.js": {
      "purpose": "Validation utilities for task data input",
      "exports": {
        "functions": [
          {
            "name": "validateTask",
            "signature": "validateTask(taskData: object) -> {isValid: boolean, error?: string}",
            "description": "Validates task data for required fields and format",
            "parameters": {
              "taskData": "Object containing task data to validate"
            },
            "returns": "Validation result with success status and optional error message"
          }
        ],
        "classes": [],
        "constants": [],
        "types": []
      },
      "imports": [],
      "sideEffects": []
    },
    "public/script.js": {
      "purpose": "Frontend JavaScript for task management UI interactions",
      "exports": {
        "functions": [
          {
            "name": "fetchTasks",
            "signature": "fetchTasks() -> Promise<Array>",
            "description": "Retrieves all tasks from the API",
            "parameters": {},
            "returns": "Promise resolving to array of task objects"
          },
          {
            "name": "createTask",
            "signature": "createTask(title: string) -> Promise<object>",
            "description": "Creates a new task via API",
            "parameters": {
              "title": "Title for the new task"
            },
            "returns": "Promise resolving to created task object"
          },
          {
            "name": "renderTasks",
            "signature": "renderTasks(tasks: Array) -> void",
            "description": "Renders task list in the DOM",
            "parameters": {
              "tasks": "Array of task objects to display"
            },
            "returns": "void"
          }
        ],
        "classes": [],
        "constants": [
          {
            "name": "API_BASE",
            "type": "string",
            "value": "/api",
            "description": "Base URL for API endpoints"
          }
        ],
        "types": []
      },
      "imports": [],
      "sideEffects": ["network-calls", "modifies-dom"]
    }
  },
  "dependencies": {
    "express": "Fast, unopinionated web framework for Node.js",
    "cors": "Middleware for enabling Cross-Origin Resource Sharing"
  },
  "architecture": {
    "main_flow": "Express server serves static files and API routes, frontend makes AJAX calls to backend",
    "data_flow": "HTTP requests -> Express routes -> Task models -> JSON responses -> Frontend rendering",
    "configuration": "Environment variables for port, package.json for dependencies",
    "entry_points": ["index.js", "public/index.html"]
  }
}
```"