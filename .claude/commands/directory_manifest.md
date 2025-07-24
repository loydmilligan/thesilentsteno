# Generate Directory Manifest Command

```bash
claude-code "Generate a codebase manifest for a specific directory with controlled depth analysis.

Start your response with: '🔍 **GENERATE_DIRECTORY_MANIFEST EXECUTING** - Analyzing directory [DIRECTORY] at depth [DEPTH]'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ DIRECTORY parameter provided (target directory path)
- ✅ DEPTH parameter provided (directory traversal depth)
- ✅ Target directory exists and is accessible
- ✅ Directory contains analyzable files
- ✅ Write permissions available for output file

**Arguments Format:**
- DIRECTORY: [target directory path - absolute or relative]
- DEPTH: [1=directory only, 2=directory+1 sublevel, 3=directory+2 sublevels, etc.]

**File Naming Convention:**
- Extract directory name from path: "src/components/ui" → "ui"
- Sanitize name: replace special characters with underscores
- Format: directory_manifest_[SANITIZED_DIRECTORY_NAME].json
- Save location: Inside the target directory itself

**Input Validation Results:**
- [ ] Directory path: [VALID/INVALID] - [DIRECTORY_PATH]
- [ ] Directory exists: [EXISTS/MISSING]
- [ ] Directory accessible: [ACCESSIBLE/PERMISSION_DENIED]
- [ ] Depth parameter: [VALID/INVALID] - [DEPTH_LEVEL]
- [ ] Files present: [FILES_FOUND/EMPTY_DIRECTORY]
- [ ] Write permissions to target directory: [AVAILABLE/DENIED]
- [ ] Path normalization: [COMPLETED/FAILED] - [NORMALIZED_PATH]
- [ ] Directory name extraction: [SUCCESS/FAILED] - [EXTRACTED_NAME]

**❌ STOP EXECUTION if any required inputs are missing or invalid**

---

## Task: Generate Directory-Specific Manifest

**Target Directory:** [DIRECTORY_PATH]
**Analysis Depth:** [DEPTH_LEVEL]
**Output File:** directory_manifest_[DIRECTORY_NAME].json (saved in target directory)

## Directory Analysis Process:

### 1. Path Normalization and Validation
- **Normalize Directory Path**: Convert to absolute path for consistency
- **Extract Directory Name**: Take final component of path (e.g., "src/components/ui" → "ui")
- **Sanitize Directory Name**: Replace spaces, slashes, special chars with underscores
- **Validate Target Access**: Ensure directory exists and is readable
- **Validate Write Access**: Ensure target directory is writable for output file
- **Define Output Path**: [TARGET_DIRECTORY]/directory_manifest_[SANITIZED_NAME].json

### 2. Directory Scope Definition
- **Target Directory**: [NORMALIZED_ABSOLUTE_PATH]
- **Depth Interpretation**:
  - `depth 1` = Only files directly in target directory (no subdirectories)
  - `depth 2` = Target directory + 1 level of subdirectories
  - `depth 3` = Target directory + 2 levels of subdirectories
- **File Types**: Include source code files (.js, .jsx, .ts, .tsx, .py, .java, .go, .rs, .php, .rb, .c, .cpp, .h, .cs, .swift, .kt)
- **Exclusions**: Skip .git/, node_modules/, dist/, build/, __pycache__/, .DS_Store, *.log, .env files

### 2. Directory Structure Mapping
- Scan target directory and subdirectories up to specified depth
- Identify all source code files within scope
- Map directory organization and file relationships
- Detect technology stack from file extensions
- Locate configuration files (package.json, requirements.txt, etc.)

### 3. File Analysis (Per Source File)
For each source code file found:
- **Purpose**: Infer file's role and responsibility
- **Exports**: Extract functions, classes, constants, interfaces
- **Imports**: Map dependencies and local file imports  
- **Side Effects**: Identify database calls, network requests, file operations

### 4. Generate Directory Manifest Structure

Create `directory_manifest_[DIRECTORY_NAME].json`:

```json
{
  \"version\": \"1.0\",
  \"generated\": \"[current timestamp in ISO format]\",
  \"analysis_scope\": {
    \"target_directory\": \"[ORIGINAL_DIRECTORY_PATH_AS_PROVIDED]\",
    \"normalized_path\": \"[NORMALIZED_ABSOLUTE_PATH]\",
    \"analysis_depth\": [DEPTH_LEVEL],
    \"directories_scanned\": [\"list of all directories included in analysis\"],
    \"total_files_analyzed\": [count],
    \"file_extensions_found\": [\"list of file extensions detected\"]
  },
  \"directory_info\": {
    \"name\": \"[directory name]\",
    \"relative_path\": \"[path relative to current working directory]\",
    \"absolute_path\": \"[full directory path]\",
    \"primary_language\": \"[most common programming language detected]\",
    \"tech_stack\": [\"technologies inferred from files and dependencies\"],
    \"file_count_by_type\": {
      \"source_files\": [count],
      \"config_files\": [count],
      \"other_files\": [count]
    }
  },
  \"files\": {
    \"[relative_file_path]\": {
      \"purpose\": \"[one line description of file's role and responsibility]\",
      \"file_type\": \"[source/config/test/documentation]\",
      \"language\": \"[programming language]\",
      \"size_lines\": \"[approximate line count]\",
      \"exports\": {
        \"functions\": [
          {
            \"name\": \"function_name\",
            \"signature\": \"function_name(param1: type, param2: type) -> return_type\",
            \"description\": \"what this function does and its purpose\",
            \"parameters\": {
              \"param1\": \"description and purpose of param1\",
              \"param2\": \"description and purpose of param2\"
            },
            \"returns\": \"description of return value and type\"
          }
        ],
        \"classes\": [
          {
            \"name\": \"ClassName\",
            \"description\": \"purpose and responsibility of this class\",
            \"constructor\": \"ClassName(param1: type, param2: type)\",
            \"methods\": [
              {
                \"name\": \"method_name\",
                \"signature\": \"method_name(param: type) -> return_type\",
                \"description\": \"what this method does and its purpose\",
                \"parameters\": {\"param\": \"parameter description\"},
                \"returns\": \"return value description\"
              }
            ],
            \"properties\": [
              {
                \"name\": \"property_name\",
                \"type\": \"property_type\",
                \"description\": \"what this property stores and represents\"
              }
            ]
          }
        ],
        \"constants\": [
          {
            \"name\": \"CONSTANT_NAME\",
            \"type\": \"constant_type\",
            \"value\": \"actual_value_or_description\",
            \"description\": \"what this constant represents and its usage\"
          }
        ],
        \"interfaces_types\": [
          {
            \"name\": \"TypeName\",
            \"definition\": \"type definition or interface structure\",
            \"description\": \"purpose and usage of this type\"
          }
        ]
      },
      \"imports\": [\"list of external packages and local file imports\"],
      \"sideEffects\": [\"writes-database\", \"reads-database\", \"network-calls\", \"writes-files\", \"reads-files\", \"creates-ui\", \"modifies-dom\", \"registers-events\"]
    }
  },
  \"dependencies\": {
    \"external_packages\": {
      \"[package-name]\": \"[brief description of what this dependency provides]\"
    },
    \"internal_dependencies\": {
      \"[file-path]\": [\"list of other files in directory that import this file\"]
    },
    \"dependency_graph\": {
      \"[file-path]\": [\"list of files this file depends on\"]
    }
  },
  \"directory_architecture\": {
    \"organization_pattern\": \"[how directory is organized - feature-based, layered, MVC, etc.]\",
    \"entry_points\": [\"main files that serve as entry points or public interfaces\"],
    \"utility_modules\": [\"helper and utility files\"],
    \"configuration_files\": [\"config files found in directory scope\"],
    \"subdirectory_purposes\": {
      \"[subdirectory-name]\": \"[purpose and role of this subdirectory]\"
    }
  }
}
```

### 5. Dependency Analysis and Mapping
- Extract external package dependencies from imports
- Map internal file dependencies within the directory scope
- Create dependency graph showing file relationships
- Identify circular dependencies if present
- Document unused files or orphaned modules

### 6. Architecture Pattern Detection
- Analyze directory organization and file naming patterns
- Identify architectural patterns (MVC, feature modules, layered, etc.)
- Determine entry points and public interfaces
- Categorize utility vs. core business logic files
- Document subdirectory purposes and organization

### 7. Quality Validation
- Verify all source files in scope were analyzed
- Ensure export analysis captured major functions and classes
- Validate dependency mapping is complete and accurate
- Confirm JSON structure follows schema
- Check file path consistency and accuracy

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ Directory manifest file created with correct naming convention
- ✅ All source files in depth scope analyzed completely
- ✅ Complete export analysis with function signatures
- ✅ Comprehensive dependency mapping internal and external
- ✅ Directory architecture patterns documented
- ✅ Valid JSON structure with all required fields

**Output Validation Results:**
- [ ] Manifest file: [CREATED/FAILED] - [FULL_PATH_TO_GENERATED_MANIFEST]
- [ ] File analysis coverage: [COMPLETE/INCOMPLETE] - [X/Y files analyzed]
- [ ] Export documentation: [COMPREHENSIVE/LACKING]
- [ ] Dependency mapping: [COMPLETE/INCOMPLETE]
- [ ] Architecture analysis: [DOCUMENTED/MISSING]
- [ ] JSON validity: [VALID/INVALID]
- [ ] File saved in target directory: [CONFIRMED/FAILED]

**✅ SUCCESS CRITERIA MET** - Directory manifest generated successfully
**❌ FAILURE** - Manifest generation incomplete or failed

## 🎯 ORCHESTRATOR REPORTING

**For Orchestrator Integration, Report These Values:**

```json
{
  \"task_completion_status\": \"[SUCCESS/FAILED]\",
  \"directory_analyzed\": \"[ORIGINAL_DIRECTORY_PATH_AS_PROVIDED]\",
  \"normalized_path\": \"[NORMALIZED_ABSOLUTE_PATH]\",
  \"output_file_path\": \"[FULL_PATH_TO_GENERATED_MANIFEST]\",
  \"directory_name_extracted\": \"[SANITIZED_DIRECTORY_NAME]\",
  \"files_analyzed_count\": [COUNT],
  \"analysis_depth_used\": [DEPTH_LEVEL],
  \"primary_language_detected\": \"[LANGUAGE]\",
  \"error_code\": \"[null_if_success/error_description_if_failed]\",
  \"execution_time_seconds\": [DURATION]
}
```

**Critical for Orchestrator:**
- **output_file_path**: Exact location where manifest was saved
- **error_code**: null for success, descriptive error for failures  
- **directory_name_extracted**: How the directory name was processed
- **task_completion_status**: Simple SUCCESS/FAILED for orchestrator logic

## 🚨 STANDARDIZED ERROR CODES

**For Orchestrator Error Handling:**
- **DIR_NOT_FOUND**: Target directory does not exist
- **DIR_NOT_ACCESSIBLE**: Cannot read target directory (permissions)
- **DIR_NOT_WRITABLE**: Cannot write to target directory
- **INVALID_DEPTH**: Depth parameter is not a valid integer
- **NO_FILES_FOUND**: Directory contains no analyzable source files
- **PATH_NORMALIZATION_FAILED**: Cannot resolve directory path
- **OUTPUT_WRITE_FAILED**: Cannot write manifest file
- **ANALYSIS_INCOMPLETE**: Partial analysis due to file access issues

## Analysis Summary Report:
- **Target Directory**: [ORIGINAL_DIRECTORY_PATH] → [NORMALIZED_ABSOLUTE_PATH]
- **Analysis Depth**: [DEPTH_LEVEL] ([X] directories scanned)
- **Files Analyzed**: [COUNT] source files
- **Technologies Detected**: [COMMA_SEPARATED_LIST]
- **Dependencies Found**: [COUNT] external, [COUNT] internal
- **Output File Path**: [FULL_PATH_TO_GENERATED_MANIFEST]
- **File Size**: [BYTES/KB]
- **Analysis Status**: [SUCCESS/FAILED]
- **Execution Time**: [SECONDS]

## Usage Examples:
```bash
# Analyze only the 'src' directory (no subdirectories)
claude-code generate_directory_manifest \"src 1\"

# Analyze 'components' directory + 1 level of subdirectories  
claude-code generate_directory_manifest \"components 2\"

# Analyze current directory + 2 levels of subdirectories
claude-code generate_directory_manifest \". 3\"

# Analyze 'api' directory + 1 sublevel
claude-code generate_directory_manifest \"api 2\"
```

The directory-specific manifest analysis is complete and saved as [FULL_PATH_TO_GENERATED_MANIFEST]."
```
