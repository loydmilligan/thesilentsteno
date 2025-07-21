# Process Task Command (Revised)

```bash
claude-code "Process and prepare a task for implementation using manifest-driven development.

Start your response with: '🔄 **PROCESS_TASK EXECUTING** - Processing [TASK_ID] for implementation'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ TASK_ID parameter provided (format: Task-X.X)
- ✅ tasks/tasks.json file exists and is readable
- ✅ Specified TASK_ID exists in tasks.json
- ✅ codebase_manifest.json exists (baseline manifest)
- ✅ Task dependencies are met (check task.dependencies array)

**Input Validation Results:**
- [ ] TASK_ID: [TASK_ID] - [VALID/INVALID]
- [ ] tasks.json: [EXISTS/MISSING]
- [ ] Task found: [FOUND/NOT_FOUND]
- [ ] Baseline manifest: [EXISTS/MISSING]
- [ ] Dependencies met: [MET/UNMET - list any missing]

**❌ STOP EXECUTION if any required inputs are missing or invalid**

---

## Task: Process Task for Implementation

**Task ID:** [TASK_ID]

## Process Steps:

### 1. Load Baseline State (MANDATORY)
- **CRITICAL:** Load existing `codebase_manifest.json` as immutable baseline
- **DO NOT** generate new manifest from codebase analysis
- **DO NOT** scan current files - use only the stored baseline
- Log baseline manifest metadata (file size, modification time, hash if available)
- If baseline doesn't exist, create minimal baseline: `{\"files\": {}, \"dependencies\": {}, \"architecture\": {}}`

### 2. Parse Task Requirements
- Read task details from `tasks/tasks.json` for the specified TASK_ID
- Extract task description, actions, acceptance criteria
- Identify files to create/modify from task definition
- Extract dependencies to add from task definition

### 3. Generate Expected Post-Task Manifest
Create detailed expected manifest showing ONLY the changes from baseline:
- **Start with baseline manifest as foundation**
- Add new files that will be created with their purpose and exports
- Update existing files with new/modified exports
- Add new dependencies that will be required
- Update architecture information for significant changes
- Document any expected side effects

### 4. Create Implementation Context
Generate comprehensive implementation notes:
- Step-by-step implementation approach
- Key technical decisions and rationale
- Integration points with existing code
- Testing requirements and validation steps
- Specific acceptance criteria for completion

### 5. Create Task Preparation File
Save complete task context as `tasks/prepared/[TASK_ID].json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"task_description\": \"[Full task description from tasks.json]\",
  \"baseline_manifest\": { [Complete LOADED baseline manifest] },
  \"expected_manifest\": { [Complete expected post-task manifest] },
  \"implementation_notes\": {
    \"approach\": \"[Step-by-step implementation approach]\",
    \"files_to_create\": [
      {
        \"file\": \"path/to/file.ext\",
        \"purpose\": \"What this file does\",
        \"key_exports\": [\"list of main exports\"]
      }
    ],
    \"files_to_modify\": [
      {
        \"file\": \"path/to/existing.ext\",
        \"changes\": \"Description of changes needed\"
      }
    ],
    \"dependencies\": [\"new dependencies to add\"],
    \"integration_points\": [\"how this connects to existing code\"],
    \"testing_approach\": \"How to test this implementation\"
  },
  \"acceptance_criteria\": [
    \"Specific criteria for task completion\"
  ],
  \"estimated_complexity\": \"[Low/Medium/High]\",
  \"prerequisites\": [\"Any tasks that must be completed first\"],
  \"baseline_metadata\": {
    \"loaded_from\": \"codebase_manifest.json\",
    \"timestamp\": \"[when baseline was loaded]\",
    \"file_count\": \"[number of files in baseline]\"
  }
}
```

### 6. Validation and Quality Checks
- Verify expected manifest builds logically from baseline
- Check for potential conflicts with existing architecture
- Ensure all dependencies are accounted for and compatible
- Validate that acceptance criteria are measurable and achievable
- Confirm implementation approach is realistic

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ Task preparation file created at tasks/prepared/[TASK_ID].json
- ✅ Baseline manifest was LOADED (not generated)
- ✅ Expected manifest contains all planned changes
- ✅ Implementation notes are comprehensive and actionable
- ✅ All task requirements are addressed in expected manifest
- ✅ No breaking changes introduced without justification

**Output Validation Results:**
- [ ] Preparation file: [CREATED/FAILED] - tasks/prepared/[TASK_ID].json
- [ ] Baseline loaded: [LOADED FROM FILE/GENERATED - ERROR]
- [ ] Expected manifest: [COMPLETE/INCOMPLETE]
- [ ] Implementation notes: [COMPREHENSIVE/LACKING]
- [ ] Task coverage: [COMPLETE/PARTIAL]
- [ ] Breaking changes: [NONE/JUSTIFIED/UNJUSTIFIED]

**✅ SUCCESS CRITERIA MET** - Task is ready for implementation
**❌ FAILURE** - Missing required outputs, do not proceed to implement_task

## Final Status Report:
- Task ID: [TASK_ID]
- Preparation file location: tasks/prepared/[TASK_ID].json
- Files to create: [count]
- Files to modify: [count]
- Dependencies to add: [count]
- Estimated complexity: [Low/Medium/High]
- Ready for implementation: [YES/NO]

The task is now prepared and ready for the implement_task command."
```