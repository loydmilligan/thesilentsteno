# Add Project Details Command

```bash
claude-code "Customize the manifest-driven development commands with project-specific details and examples.

## Task: Customize Commands for Current Project

**Purpose:** Analyze the current project and update command files with relevant examples, side effects, and project-specific guidance.

Start your response with: "🎨 **ADD_PROJECT_DETAILS EXECUTING** - Customizing commands for current project"

## Customization Process:

### 1. Analyze Current Project
- Read `codebase_manifest.json` to understand project type and stack
- Read `package.json` to identify dependencies and framework
- Scan existing files to determine architecture patterns
- Identify primary programming language and frameworks

### 2. Determine Project Profile
Based on analysis, classify project as:
- **Web Application** (Express.js, React, Vue.js, etc.)
- **Content Management System** (CMS, blog, documentation)
- **API Service** (REST API, GraphQL, microservice)
- **Desktop Application** (Electron, Tauri, native)
- **Mobile Application** (React Native, Flutter, etc.)
- **CLI Tool** (Command-line utility, scripts)
- **Library/Package** (NPM package, component library)
- **Browser Extension** (Chrome extension, web extension)
- **Other** (Custom classification based on analysis)

### 3. Generate Project-Specific Context
Create project context including:
- **Tech Stack**: Primary technologies and frameworks
- **Common Side Effects**: Relevant side effects for this project type
- **Example Commit Messages**: Realistic examples for this project
- **Typical File Patterns**: Common file types and structures
- **Integration Points**: Likely external services and APIs

### 4. Update Command Files
For each command in `.claude/commands/`, update with project-specific content:

#### **generate_manifest.md**
- Add project-specific side effects to the categories list
- Include examples relevant to the tech stack
- Update file type scanning for project's languages

#### **commit_task.md**
- Replace generic examples with project-appropriate commit messages
- Add project-specific commit message patterns

#### **implement_task.md**
- Add project-specific coding standards and conventions
- Include framework-specific implementation guidance

#### **check_task.md** 
- Add project-specific validation criteria
- Include tech stack specific testing approaches

#### **process_task.md**
- Add project-specific analysis patterns