# Claude Code Manifest Generator Prompt

```bash
claude-code "Please analyze the current codebase and generate a comprehensive manifest file following these specifications:

## Task: Generate codebase_manifest.json

Create a manifest file that maps the current state of the codebase. Scan all files and analyze their structure to create an accurate representation.

Start your response with: "🔍 **GENERATE_MANIFEST EXECUTING** - Analyzing current codebase and creating manifest"

## Manifest Format Required:

```json
{
  \"version\": \"1.0\",
  \"generated\": \"[current timestamp in ISO format]\",
  \"project\": \"[infer project name from package.json or directory]\",
  \"description\": \"[brief description of what this codebase does]\",
  \"files\": {
    \"path/to/file.ext\": {
      \"purpose\": \"[one line description of what this file does]\",
      \"exports\": {
        \"functions\": [
          {
            \"name\": \"function_name\",
            \"signature\": \"function_name(param1: type, param2: type) -> return_type\",
            \"description\": \"what this function does\",
            \"parameters\": {
              \"param1\": \"description of param1\",
              \"param2\": \"description of param2\"
            },
            \"returns\": \"description of return value\"
          }
        ],
        \"classes\": [
          {
            \"name\": \"ClassName\",
            \"description\": \"what this class does\",
            \"constructor\": \"ClassName(param1: type, param2: type)\",
            \"methods\": [
              {
                \"name\": \"method_name\",
                \"signature\": \"method_name(param: type) -> return_type\",
                \"description\": \"what this method does\",
                \"parameters\": {\"param\": \"description\"},
                \"returns\": \"description of return value\"
              }
            ],
            \"properties\": [
              {
                \"name\": \"property_name\",
                \"type\": \"property_type\",
                \"description\": \"what this property stores\"
              }
            ]
          }
        ],
        \"constants\": [
          {
            \"name\": \"CONSTANT_NAME\",
            \"type\": \"constant_type\",
            \"value\": \"actual_value_or_description\",
            \"description\": \"what this constant represents\"
          }
        ]
      },
      \"imports\": [\"list of main dependencies and local imports\"],
      \"sideEffects\": [\"list of side effects like 'writes-database', 'network-calls', 'modifies-files', 'creates-ui', etc.\"]
    }
  },
  \"dependencies\": {
    \"[package-name]\": \"[brief description of what this dependency provides]\"
  },
  \"architecture\": {
    \"main_flow\": \"[describe the main execution flow]\",
    \"data_flow\": \"[describe how data flows through the system]\",
    \"configuration\": \"[describe how the system is configured]\"
  }
}
```

## Analysis Instructions:

1. **Scan all files** in the current directory and subdirectories
2. **Ignore** these files/directories:
   - node_modules/
   - .git/
   - dist/
   - build/
   - .DS_Store
   - *.log files
   - .env files (but note if they exist)

3. **For each file, determine:**
   - **Purpose**: What does this file do? (one concise sentence)
   - **Exports**: What functions, classes, constants, or types does it export? Include full API details.
   - **Imports**: What external packages and local files does it import?
   - **Side Effects**: What does it do beyond pure computation?

4. **For exports, provide complete API documentation:**
   - **Functions**: Include signature, parameters, return values, and description
   - **Classes**: Include constructor, all public methods, properties, and descriptions
   - **Constants**: Include type, value, and purpose
   - **Method details**: Include parameter types, return types, and what each method does

4. **Side Effects Categories:**
   - 'writes-database' - modifies persistent storage
   - 'reads-database' - reads from persistent storage  
   - 'network-calls' - makes HTTP/API calls
   - 'writes-files' - creates or modifies files
   - 'reads-files' - reads from files
   - 'creates-ui' - creates user interface elements
   - 'modifies-dom' - changes DOM elements
   - 'registers-events' - sets up event listeners
   - 'registers-commands' - adds commands to systems
   - 'connects-services' - establishes external service connections
   - 'publishes-messages' - sends messages to queues or services
   - 'subscribes-events' - listens for external events
   - 'loads-settings' - reads configuration
   - 'saves-settings' - writes configuration

5. **For package.json dependencies**, read the actual dependencies and provide brief descriptions of what each major dependency does.

6. **Architecture Analysis:**
   - Identify the main entry point
   - Trace the primary execution flow
   - Identify how data moves through the system
   - Note how configuration is handled

## Output Requirements:

- Create the file as `codebase_manifest.json` in the root directory
- Use proper JSON formatting with proper escaping
- Include all files that contain actual code (not just config files)
- Be accurate about exports - read the actual export statements
- Be accurate about imports - read the actual import statements
- If a file doesn't exist yet but is referenced, note it in the manifest with \"status\": \"planned\"

## Example Analysis Process:

If analyzing a simple web application:

```
project/
├── app.js
├── routes/
│   ├── index.js
│   └── api.js
├── models/
│   └── user.js
├── package.json
└── README.md
```

With these file contents:

**app.js:**
```javascript
const express = require('express');
const routes = require('./routes');

const app = express();
app.use('/api', routes);

module.exports = app;
```

**routes/index.js:**
```javascript
const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  res.json({ message: 'Welcome to API' });
});

module.exports = router;
```

The analysis would produce:

1. **Read app.js** - imports express, routes; exports app; side effects: creates-server
2. **Read routes/index.js** - imports express; exports router; side effects: registers-routes
3. **Skip README.md** - documentation file

**Expected Generated Manifest:**

```json
{
  \"version\": \"1.0\",
  \"generated\": \"2025-07-08T15:30:00Z\",
  \"project\": \"example-web-app\",
  \"description\": \"A simple Express.js web application with API routes\",
  \"files\": {
    \"app.js\": {
      \"purpose\": \"Main application entry point that configures Express server\",
      \"exports\": {
        \"functions\": [],
        \"classes\": [],
        \"constants\": [
          {
            \"name\": \"app\",
            \"type\": \"Express.Application\",
            \"value\": \"Express application instance\",
            \"description\": \"Configured Express application ready to be started\"
          }
        ]
      },
      \"imports\": [\"express\", \"./routes\"],
      \"sideEffects\": [\"creates-server\", \"registers-routes\"]
    },
    \"routes/index.js\": {
      \"purpose\": \"Defines API routes and endpoints\",
      \"exports\": {
        \"functions\": [],
        \"classes\": [],
        \"constants\": [
          {
            \"name\": \"router\",
            \"type\": \"Express.Router\",
            \"value\": \"Express router instance\",
            \"description\": \"Router containing API endpoint definitions\"
          }
        ]
      },
      \"imports\": [\"express\"],
      \"sideEffects\": [\"registers-routes\"]
    }
  },
  \"dependencies\": {
    \"express\": \"Web framework for Node.js applications\"
  },
  \"architecture\": {
    \"main_flow\": \"app.js creates Express server -> routes/index.js defines endpoints -> server ready to handle requests\",
    \"data_flow\": \"HTTP requests -> Express router -> route handlers -> JSON responses\",
    \"configuration\": \"Configuration handled through Express middleware and route setup\"
  }
}
```

Please analyze the codebase thoroughly and create an accurate manifest that represents the current state of the project."
```