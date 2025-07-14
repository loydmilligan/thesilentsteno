# Commit Task Command (Revised)

```bash
claude-code "Commit completed task implementation with proper git history and manifest updates.

Start your response with: '💾 **COMMIT_TASK EXECUTING** - Committing [TASK_ID] implementation'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ TASK_ID parameter provided (format: Task-X.X)
- ✅ tasks/validation/[TASK_ID]-validation.json exists
- ✅ Validation status is PASS or MINOR_ISSUES
- ✅ Git repository is initialized and accessible
- ✅ Working directory is clean or changes are task-related
- ✅ codebase_manifest.json exists (to be updated)

**Input Validation Results:**
- [ ] TASK_ID: [TASK_ID] - [VALID/INVALID]
- [ ] Validation report: [EXISTS/MISSING] - tasks/validation/[TASK_ID]-validation.json
- [ ] Validation status: [PASS/MINOR_ISSUES/MAJOR_ISSUES] - [ACCEPTABLE/UNACCEPTABLE]
- [ ] Git repository: [INITIALIZED/NOT_INITIALIZED]
- [ ] Working directory: [CLEAN/HAS_CHANGES] - [TASK_RELATED/UNRELATED]
- [ ] Baseline manifest: [EXISTS/MISSING] - codebase_manifest.json

**❌ STOP EXECUTION if validation failed or critical inputs missing**

---

## Task: Commit Task Implementation

**Task ID:** [TASK_ID]

## Commit Process:

### 1. Pre-Commit Verification
- Load validation report to confirm task completion status
- Verify all critical and concerning issues have been resolved
- Ensure no uncommitted debugging or temporary code remains
- Confirm all task-related files are ready for commit
- Check that build/compile succeeds if applicable

### 2. Update Baseline Manifest
**CRITICAL STEP:** Update the project's baseline manifest with actual state:
- Load the actual_manifest from validation report
- **Replace** codebase_manifest.json content with actual_manifest
- This establishes the new baseline for future tasks
- Verify the updated manifest is valid JSON/format
- Include manifest update in the commit

### 3. Stage All Task-Related Changes
Identify and stage all files changed during task implementation:
- New files created during implementation
- Modified files changed during implementation  
- Updated dependency files (package.json, etc.)
- Configuration files modified
- Documentation updates
- **Updated codebase_manifest.json**

Review staged changes to ensure only task-related modifications are included.

### 4. Create Descriptive Commit Message

**Format:** `[TASK_ID]: [Brief description]`

**Examples:**
- `Task-1.1: Initialize project structure and build system`
- `Task-2.3: Implement user authentication service`
- `Task-4.2: Add data validation and error handling`

### 5. Create Detailed Commit Body

```
[TASK_ID]: [Brief description]

Implementation Summary:
- [Key change 1]
- [Key change 2] 
- [Key change 3]

Files Created:
- [new_file1.ext] - [purpose]
- [new_file2.ext] - [purpose]

Files Modified:
- [existing_file1.ext] - [what changed]
- [existing_file2.ext] - [what changed]

Dependencies Added:
- [dependency1] - [reason]
- [dependency2] - [reason]

Validation Status: [PASS/MINOR_ISSUES]
Breaking Changes: [YES/NO - describe if yes]
Baseline Manifest Updated: YES
Task Completion: [percentage or status]

[Any important notes or warnings for future developers]
```

### 6. Execute Git Commit
```bash
git add .
git commit -m \"[TASK_ID]: [Brief description]\" -m \"[Detailed body]\"
```

### 7. Post-Commit Tasks
- Record commit hash for task tracking
- Move task preparation file to completed directory
- Update task status in tracking system
- Archive validation report with commit reference
- Note any lessons learned or process improvements

### 8. Update Task Tracking

Move and update task files:
- Move `tasks/prepared/[TASK_ID].json` to `tasks/completed/[TASK_ID].json`
- Add commit information to completed task file:
  ```json
  {
    ...existing task data...,
    \"completion\": {
      \"status\": \"completed\",
      \"commit_hash\": \"[full commit hash]\",
      \"commit_timestamp\": \"[ISO timestamp]\",
      \"validation_status\": \"[PASS/MINOR_ISSUES]\",
      \"lessons_learned\": [\"any insights from implementation\"]
    }
  }
  ```

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ codebase_manifest.json updated with actual state
- ✅ All task-related changes committed to git
- ✅ Commit message follows proper format
- ✅ Commit includes all necessary files
- ✅ Task files moved to completed directory
- ✅ Commit hash recorded in task tracking
- ✅ New baseline established for future tasks

**Output Validation Results:**
- [ ] Manifest update: [COMPLETED/FAILED] - codebase_manifest.json
- [ ] Git commit: [SUCCESSFUL/FAILED] - [commit hash]
- [ ] Commit message: [PROPER FORMAT/INCORRECT FORMAT]
- [ ] File inclusion: [ALL INCLUDED/MISSING FILES]
- [ ] Task archival: [COMPLETED/FAILED] - tasks/completed/[TASK_ID].json
- [ ] Commit recording: [RECORDED/FAILED] in task tracking
- [ ] Baseline establishment: [SUCCESSFUL/FAILED]

**✅ SUCCESS CRITERIA MET** - Task committed, baseline updated, ready for next task
**❌ FAILURE** - Commit incomplete or baseline not updated properly

## Commit Summary Report:
- Task ID: [TASK_ID]
- Commit Hash: [full commit hash]
- Commit Timestamp: [ISO timestamp]
- Files Created: [count] - [list]
- Files Modified: [count] - [list]  
- Dependencies Added: [count] - [list]
- Baseline Manifest: [UPDATED/FAILED]
- Task Status: [COMPLETED/FAILED]
- Breaking Changes: [YES/NO - details if yes]
- Ready for Next Task: [YES/NO]

## Project Status:
- New baseline manifest established: [YES/NO]
- Git history updated: [YES/NO] 
- Task properly archived: [YES/NO]
- Development can continue with next task: [YES/NO]

The task is now complete and committed. The updated baseline manifest is ready for the next development task."
```