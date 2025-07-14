# Implement Task Command (Revised)

```bash
claude-code "Implement a prepared task with full context using manifest-driven development.

Start your response with: '⚙️ **IMPLEMENT_TASK EXECUTING** - Implementing [TASK_ID]'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ TASK_ID parameter provided (format: Task-X.X)
- ✅ tasks/prepared/[TASK_ID].json exists and is readable
- ✅ Preparation file contains valid baseline_manifest
- ✅ Preparation file contains valid expected_manifest
- ✅ Implementation notes are comprehensive
- ✅ Current codebase state matches baseline expectations

**Input Validation Results:**
- [ ] TASK_ID: [TASK_ID] - [VALID/INVALID]
- [ ] Preparation file: [EXISTS/MISSING] - tasks/prepared/[TASK_ID].json
- [ ] Baseline manifest: [VALID/INVALID/MISSING]
- [ ] Expected manifest: [VALID/INVALID/MISSING]
- [ ] Implementation notes: [COMPREHENSIVE/INCOMPLETE]
- [ ] Codebase state: [MATCHES BASELINE/DIVERGED]

**❌ STOP EXECUTION if any required inputs are missing or invalid**

---

## Task: Implement Prepared Task

**Task File:** tasks/prepared/[TASK_ID].json

## Implementation Process:

### 1. Load Task Context
- Read the complete task preparation file
- Parse baseline manifest (the starting state)
- Parse expected manifest (the target state)
- Review implementation notes and approach
- Understand acceptance criteria

### 2. Verify Prerequisites and Environment
- Confirm all prerequisite tasks are completed
- Verify current codebase state aligns with baseline manifest
- Check that required tools and dependencies are available
- Validate file paths and directory structure exist

### 3. Implementation Phase

#### Create New Files:
For each file in `files_to_create`:
- Create file with proper directory structure (create directories if needed)
- Implement all planned exports with full functionality
- Add proper imports as specified in expected manifest
- Include comprehensive comments and documentation
- Follow established coding standards and conventions
- Add proper error handling and validation

#### Modify Existing Files:
For each file in `files_to_modify`:
- Load current file content
- Make specified changes while preserving existing functionality
- Update exports as planned in expected manifest
- Maintain backward compatibility where possible
- Update imports if dependencies change
- Preserve existing comments and documentation where appropriate

#### Add Dependencies:
- Update package.json or equivalent dependency file
- Install/update packages as needed
- Update configuration files (tsconfig.json, build configs, etc.)
- Verify compatibility with existing dependencies
- Test that new dependencies load correctly

### 4. Integration and Basic Testing

#### Code Integration:
- Ensure all imports resolve correctly
- Verify function signatures match expected manifest
- Test integration points with existing code
- Handle error cases and edge conditions appropriately
- Verify no circular dependencies introduced

#### Functional Testing:
- Verify code compiles/builds without errors
- Test basic functionality of new features
- Ensure existing functionality still works (no regressions)
- Run available test suites if they exist
- Test error handling and edge cases

### 5. Code Quality and Standards

#### Quality Checks:
- Proper error handling throughout
- Consistent coding style with existing codebase
- Adequate comments and documentation
- No debugging code left behind (console.log, etc.)
- Proper typing (TypeScript) or documentation (other languages)
- Follow security best practices

#### Documentation:
- Update relevant documentation files
- Add inline documentation for complex functions/classes
- Document any assumptions or limitations
- Note any deviations from the original plan with rationale

### 6. Implementation Validation

#### Against Expected Manifest:
- Verify all planned exports are implemented correctly
- Check that file purposes match actual implementation
- Ensure side effects are as expected in manifest
- Validate integration points work as planned
- Confirm dependency additions are complete

#### Final Verification:
- All planned functionality is working
- No breaking changes to existing functionality
- Implementation meets all acceptance criteria
- Code is ready for validation phase
- All temporary/debugging artifacts removed

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ All planned files created with correct content
- ✅ All planned file modifications completed
- ✅ All planned dependencies added successfully
- ✅ Code compiles/builds without errors
- ✅ Basic functionality testing passes
- ✅ No regressions in existing functionality
- ✅ Implementation matches expected manifest structure

**Output Validation Results:**
- [ ] Files created: [COUNT/EXPECTED COUNT] - [ALL CREATED/MISSING FILES]
- [ ] Files modified: [COUNT/EXPECTED COUNT] - [ALL MODIFIED/MISSING CHANGES]
- [ ] Dependencies added: [COUNT/EXPECTED COUNT] - [ALL ADDED/MISSING DEPS]
- [ ] Build status: [SUCCESS/FAILED] - [ERROR DETAILS IF FAILED]
- [ ] Basic testing: [PASSED/FAILED] - [ISSUE DETAILS IF FAILED]
- [ ] Regression check: [NO REGRESSIONS/REGRESSIONS FOUND]
- [ ] Manifest alignment: [MATCHES EXPECTED/DEVIATIONS FOUND]

**✅ SUCCESS CRITERIA MET** - Implementation complete, ready for validation
**❌ FAILURE** - Implementation incomplete or errors found, do not proceed to check_task

## Implementation Summary Report:
- Task ID: [TASK_ID]
- Files created: [list of created files]
- Files modified: [list of modified files]
- Dependencies added: [list of new dependencies]
- Key functionality implemented: [summary of main features]
- Known deviations from plan: [any changes made during implementation]
- Build status: [SUCCESS/FAILED]
- Basic testing results: [PASSED/FAILED with details]
- Ready for validation: [YES/NO]

## Next Steps:
The implementation is complete and ready for validation using the check_task command."
```