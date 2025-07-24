# Orchestrate Directory Manifest Command

```bash
claude-code "Orchestrate parallel sub-agents to generate directory manifests across a repository structure.

Start your response with: '🗂️ **ORCHESTRATE_DIRECTORY_MANIFEST EXECUTING** - Mapping repository structure with [FILTER_STRATEGY]'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ FILTER_STRATEGY parameter provided (filtering approach)
- ✅ Current directory is project root or valid starting point
- ✅ Repository structure is accessible and readable
- ✅ Write permissions available in target directories
- ✅ Sub-agent commands are available

**Arguments Format:**
- FILTER_STRATEGY: [all/dev/full-stack/core] OR [exclude:category1,category2] OR [include:category1,category2]

**Preset Filter Levels:**
- **`all`**: Every directory regardless of content
- **`dev`**: Development-related (excludes docs, assets)
- **`full-stack`**: Main code + backend/database directories
- **`core`**: Just main source code directories

**Flexible Filter Examples:**
- **`exclude:docs,assets,tests`**: Skip documentation, assets, and test directories
- **`include:src,lib,api,components`**: Only analyze specified directory types

**Input Validation Results:**
- [ ] Filter strategy: [VALID/INVALID] - [FILTER_STRATEGY]
- [ ] Repository access: [ACCESSIBLE/DENIED]
- [ ] Starting directory: [VALID_PROJECT_ROOT/INVALID]
- [ ] Write permissions: [AVAILABLE/DENIED]
- [ ] Sub-agent availability: [CONFIRMED/MISSING]

**❌ STOP EXECUTION if any required inputs are missing or invalid**

---

## Task: Orchestrate Directory Manifest Generation

**Filter Strategy:** [FILTER_STRATEGY]
**Repository Root:** [CURRENT_DIRECTORY]
**Max Depth:** 3 levels from root (root + 2 sublevels)
**Batch Size:** 4 parallel sub-agents
**Max Directories:** 20 total (safety limit)

## Orchestration Process:

### 1. Repository Structure Analysis

**Directory Tree Scanning:**
- Scan repository from root to max depth of 3 levels
- Identify all directories containing source code files
- Categorize directories by type and content
- Apply filtering strategy to determine target directories
- Exclude empty directories (no source files)

**Directory Classification System:**
```
Category Definitions:
- docs: documentation/, docs/, README/, wiki/, guides/
- assets: assets/, static/, images/, media/, public/, fonts/
- tests: test/, tests/, __tests__/, spec/, cypress/, e2e/
- config: config/, .github/, scripts/, build/, dist/, node_modules/
- src: src/, lib/, source/, app/, main/
- components: components/, widgets/, ui/, views/, pages/
- api: api/, server/, backend/, services/, controllers/
- database: database/, db/, migrations/, models/, schema/
- examples: examples/, demo/, samples/, playground/
- templates: templates/, layouts/, themes/
```

**Filter Application Logic:**
```
IF filter_strategy == "all":
    INCLUDE all directories with source files
ELIF filter_strategy == "dev":
    EXCLUDE docs, assets
    INCLUDE config, src, components, api, database, examples, templates, tests
ELIF filter_strategy == "full-stack":
    INCLUDE src, components, api, database
    EXCLUDE docs, assets, config, examples, templates, tests
ELIF filter_strategy == "core":
    INCLUDE src, components
    EXCLUDE all others
ELIF filter_strategy.startswith("exclude:"):
    EXCLUDE specified categories
    INCLUDE all others
ELIF filter_strategy.startswith("include:"):
    INCLUDE only specified categories
    EXCLUDE all others
```

### 2. Target Directory Validation

**Directory Analysis:**
- Verify each target directory exists and is accessible
- Confirm directories contain analyzable source files
- Check write permissions for manifest output
- Estimate total sub-agent deployment count
- Apply safety limit (max 20 directories)

**Safety Checks:**
- If > 20 directories: Report count and request confirmation
- If any directories inaccessible: Report and skip
- If no qualifying directories found: Report and exit

### 3. Sub-Agent Deployment Strategy

**Parallel Batch Coordination:**
- Deploy sub-agents in batches of 4 simultaneously using Task tool
- Launch all 4 sub-agents in a batch at the same time (true parallel execution)
- Monitor batch completion - wait for ALL 4 to complete before next batch
- Handle sub-agent failures by skipping failed directories
- Track progress and provide status updates throughout parallel execution

**Sub-Agent Parallel Deployment Protocol:**
Each batch deploys 4 sub-agents simultaneously:
```
PARALLEL_BATCH_DEPLOYMENT:
  Launch Task 1: generate_directory_manifest "[DIR1] 1"
  Launch Task 2: generate_directory_manifest "[DIR2] 1" 
  Launch Task 3: generate_directory_manifest "[DIR3] 1"
  Launch Task 4: generate_directory_manifest "[DIR4] 1"
  
  WAIT for all 4 tasks to complete
  COLLECT results from all 4 sub-agents
  PROCEED to next batch only after all complete
```

**Sub-Agent Task Assignment:**
Each sub-agent receives these complete instructions:

```
TASK: Generate directory manifest for [TARGET_DIRECTORY]

🔍 **GENERATE_DIRECTORY_MANIFEST EXECUTING** - Analyzing directory [TARGET_DIRECTORY] at depth 1

You are Sub-Agent [X] analyzing directory: [TARGET_DIRECTORY]

CRITICAL INSTRUCTIONS:
- Use EXACTLY these arguments: "[TARGET_DIRECTORY] 1"
- Depth is ALWAYS 1 (directory only, no subdirectories)
- Save manifest file IN the target directory itself
- Report back with structured JSON response

ARGUMENTS: "[TARGET_DIRECTORY] 1"

Your task is to execute the generate_directory_manifest command with the directory path and depth 1.

EXPECTED OUTPUT:
- File saved as: [TARGET_DIRECTORY]/directory_manifest_[DIR_NAME].json
- Structured reporting with file path and completion status
- Error reporting if directory cannot be analyzed

Execute the command and report results.
```

### 4. Batch Execution Management

**Batch 1 Deployment (First 4 directories):**
Deploy 4 sub-agents simultaneously using Task tool:
```
Task 1: claude-code generate_directory_manifest "[TARGET_DIR_1] 1"
Task 2: claude-code generate_directory_manifest "[TARGET_DIR_2] 1"
Task 3: claude-code generate_directory_manifest "[TARGET_DIR_3] 1"
Task 4: claude-code generate_directory_manifest "[TARGET_DIR_4] 1"
```
- Launch all 4 tasks in parallel immediately
- Monitor completion and collect results from all 4 sub-agents
- Do NOT proceed to next batch until all 4 complete

**Subsequent Batch Deployment:**
- Wait for ALL previous batch sub-agents to complete
- Deploy next batch of 4 sub-agents simultaneously using Task tool
- Continue parallel deployment until all target directories processed
- Handle any sub-agent failures by logging and continuing with remaining batches

**Progress Tracking:**
```
🎯 **ORCHESTRATION_PROGRESS**
Batch [X] of [Y]: Deploying 4 sub-agents in parallel
Launching Tasks: [DIR1], [DIR2], [DIR3], [DIR4]
Directories remaining: [COUNT]
Completed: [COUNT] | Failed: [COUNT] | In Progress: 4
Waiting for batch completion before next deployment...
```

**CRITICAL: True Parallel Execution**
The orchestrator MUST use the Task tool to spawn multiple sub-agents simultaneously, not sequentially. Each batch launches all 4 sub-agents at exactly the same time for maximum efficiency.

**Example Parallel Execution:**
```
Found 10 target directories: src/, api/, components/, database/, config/, utils/, types/, hooks/, services/, tests/

BATCH 1 - Deploy 4 simultaneously:
  🚀 Task 1: claude-code generate_directory_manifest "src 1"
  🚀 Task 2: claude-code generate_directory_manifest "api 1"  
  🚀 Task 3: claude-code generate_directory_manifest "components 1"
  🚀 Task 4: claude-code generate_directory_manifest "database 1"
  
  ⏳ Wait for ALL 4 to complete...
  ✅ Collect results from all 4 sub-agents
  
BATCH 2 - Deploy next 4 simultaneously:
  🚀 Task 5: claude-code generate_directory_manifest "config 1"
  🚀 Task 6: claude-code generate_directory_manifest "utils 1"
  🚀 Task 7: claude-code generate_directory_manifest "types 1" 
  🚀 Task 8: claude-code generate_directory_manifest "hooks 1"
  
  ⏳ Wait for ALL 4 to complete...
  ✅ Collect results from all 4 sub-agents

BATCH 3 - Deploy remaining 2 simultaneously:
  🚀 Task 9: claude-code generate_directory_manifest "services 1"
  🚀 Task 10: claude-code generate_directory_manifest "tests 1"
  
  ⏳ Wait for both to complete...
  ✅ Collect results and finalize orchestration
```

### 5. Result Collection and Validation

**Sub-Agent Result Processing:**
For each completed sub-agent, collect:
- Task completion status (SUCCESS/FAILED)
- Output file path (where manifest was saved)
- Directory analyzed and file count
- Error details (if failed)
- Execution time

**Quality Validation:**
- Verify manifest files were created in expected locations
- Validate JSON structure of generated manifests
- Check file accessibility and readability
- Report any inconsistencies or missing files

### 6. Comprehensive Summary Generation

**Orchestration Summary Report:**
```json
{
  "orchestration_metadata": {
    "execution_timestamp": "[ISO timestamp]",
    "filter_strategy": "[FILTER_STRATEGY]",
    "repository_root": "[ABSOLUTE_PATH_TO_ROOT]",
    "max_depth_analyzed": 3,
    "batch_size": 4,
    "total_directories_found": [COUNT],
    "total_directories_analyzed": [COUNT],
    "total_sub_agents_deployed": [COUNT],
    "execution_duration_seconds": [DURATION]
  },
  "directory_analysis_results": [
    {
      "directory_path": "[RELATIVE_PATH]",
      "directory_category": "[src/api/docs/etc]",
      "analysis_status": "[SUCCESS/FAILED/SKIPPED]",
      "manifest_file_path": "[FULL_PATH_TO_MANIFEST]",
      "files_analyzed": [COUNT],
      "primary_language": "[DETECTED_LANGUAGE]",
      "error_details": "[null_or_error_description]"
    }
  ],
  "filtering_summary": {
    "directories_included": [COUNT],
    "directories_excluded_by_filter": [COUNT],
    "directories_skipped_empty": [COUNT],
    "categories_included": ["list of included categories"],
    "categories_excluded": ["list of excluded categories"]
  },
  "success_metrics": {
    "successful_analyses": [COUNT],
    "failed_analyses": [COUNT],
    "success_rate_percentage": [PERCENTAGE],
    "total_manifest_files_created": [COUNT],
    "total_source_files_analyzed": [COUNT]
  }
}
```

**Human-Readable Summary:**
```
🗂️ **REPOSITORY MANIFEST ORCHESTRATION COMPLETE**

📊 **Execution Summary:**
- Filter Strategy: [FILTER_STRATEGY]
- Directories Analyzed: [SUCCESS_COUNT] of [TOTAL_COUNT]
- Sub-Agents Deployed: [AGENT_COUNT] in [BATCH_COUNT] batches
- Execution Time: [DURATION]
- Success Rate: [PERCENTAGE]%

📁 **Directory Breakdown:**
- Source Code: [COUNT] directories
- API/Backend: [COUNT] directories  
- Configuration: [COUNT] directories
- Documentation: [COUNT] directories (if included)
- Other: [COUNT] directories

✅ **Generated Manifests:**
[List of all created manifest files with paths]

❌ **Failed Analyses:**
[List of failed directories with error reasons]

🎯 **Next Steps:**
- Review generated manifests in their respective directories
- Use master manifest generator to combine all directory manifests
- Validate manifest completeness and accuracy
```

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ Repository structure analyzed and categorized
- ✅ Filtering strategy applied successfully
- ✅ All target directories processed by sub-agents
- ✅ Directory manifests created in appropriate locations
- ✅ Comprehensive orchestration summary generated
- ✅ Error handling completed for failed sub-agents

**Output Validation Results:**
- [ ] Structure analysis: [COMPLETE/INCOMPLETE]
- [ ] Filter application: [SUCCESSFUL/FAILED]
- [ ] Sub-agent deployment: [ALL_DEPLOYED/PARTIAL]
- [ ] Manifest generation: [COMPLETE/PARTIAL] - [X/Y successful]
- [ ] Summary report: [GENERATED/MISSING]
- [ ] Error documentation: [COMPLETE/INCOMPLETE]

**✅ SUCCESS CRITERIA MET** - Repository manifest orchestration completed
**❌ FAILURE** - Orchestration incomplete, review error details

## Orchestration Results:
- **Repository Root**: [ABSOLUTE_PATH]
- **Filter Strategy**: [FILTER_STRATEGY]
- **Directories Processed**: [SUCCESS_COUNT] of [TOTAL_COUNT]
- **Manifest Files Created**: [COUNT]
- **Sub-Agents Deployed**: [COUNT] in [BATCH_COUNT] batches
- **Success Rate**: [PERCENTAGE]%
- **Total Execution Time**: [DURATION]
- **Summary Report**: orchestration_summary.json

## Usage Examples:
```bash
# Analyze all development directories (exclude docs/assets)
claude-code orchestrate_directory_manifest "dev"

# Analyze only core source code directories
claude-code orchestrate_directory_manifest "core"

# Custom exclusion of docs, assets, and tests
claude-code orchestrate_directory_manifest "exclude:docs,assets,tests"

# Include only specific directory types
claude-code orchestrate_directory_manifest "include:src,api,components"

# Analyze everything (use with caution)
claude-code orchestrate_directory_manifest "all"
```

## Error Handling and Recovery:

**Common Issues:**
- **Too Many Directories**: If > 20 directories found, orchestrator will report count and request confirmation
- **Sub-Agent Failures**: Failed sub-agents are logged and skipped, orchestration continues
- **Permission Issues**: Directories without write access are skipped with error reporting
- **Empty Directories**: Directories with no source files are automatically skipped

**Recovery Strategies:**
- **Partial Success**: Orchestrator completes successfully even if some sub-agents fail
- **Retry Logic**: Failed directories can be re-run individually using generate_directory_manifest
- **Progress Preservation**: Successfully generated manifests are preserved even if later batches fail

The repository manifest orchestration is complete. All qualifying directories have been analyzed and directory manifests are available in their respective locations."
```
