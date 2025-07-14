# Check Task Command

```bash
claude-code "Verify task implementation matches expected changes using manifest comparison.

## Task: Validate Task Implementation

**Task ID:** [TASK_ID - e.g., Task-1.1]

Start your response with: "✅ **CHECK_TASK EXECUTING** - Validating [TASK_ID] implementation"

## Validation Process:

### 1. Load Task Context
- Read the prepared task file from `tasks/prepared/[TASK_ID].json`
- Extract the expected post-task manifest
- Understand what changes were supposed to be made

### 2. Generate Current Manifest
- Use the `generate_manifest` command to create a fresh manifest from the current codebase
- This reflects the actual state after implementation

### 3. Compare Manifests

#### File-Level Comparison:
- **New Files:** Check if all planned files were created
- **Modified Files:** Verify expected changes were made
- **Unexpected Files:** Identify any files created that weren't planned

#### Export-Level Comparison:
For each file, compare:
- **Functions:** Signatures, parameters, return types
- **Classes:** Constructor, methods, properties
- **Constants:** Names, types, values
- **Interfaces:** Structure and properties (if applicable)

#### Dependency Comparison:
- **New Dependencies:** Verify all planned dependencies were added
- **Missing Dependencies:** Check if any planned dependencies are missing
- **Unexpected Dependencies:** Identify any unplanned dependencies

#### Architecture Comparison:
- **Main Flow:** Check if architectural changes match expectations
- **Data Flow:** Verify data flow updates
- **Integration Points:** Confirm integration points work as planned
- **Side Effects:** Validate side effects match expectations

### 4. Categorize Differences

#### Exact Match:
- Implementation perfectly matches expected manifest
- All planned changes implemented correctly
- No unexpected changes

#### Acceptable Variations:
- Minor implementation details that don't affect API
- Better error handling than planned
- More comprehensive validation
- Additional helpful comments/documentation

#### Problematic Differences:
- **Missing Exports:** Planned functions/classes not implemented
- **Changed Signatures:** Function signatures differ from expected
- **Missing Files:** Planned files not created
- **Unexpected Breaking Changes:** Changes that break existing functionality
- **Scope Creep:** Implementation beyond task scope

### 5. Generate Comparison Report

Create detailed report in `tasks/validation/[TASK_ID]-comparison.json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"validation_timestamp\": \"[timestamp]\",
  \"overall_status\": \"[MATCH/MINOR_DIFFERENCES/MAJOR_DIFFERENCES]\",
  \"summary\": {
    \"files_created\": [\"list of files created\"],
    \"files_modified\": [\"list of files modified\"],
    \"exports_added\": [\"list of new exports\"],
    \"dependencies_added\": [\"list of new dependencies\"]
  },
  \"differences\": {
    \"missing_files\": [\"files that should have been created\"],
    \"unexpected_files\": [\"files created that weren't planned\"],
    \"missing_exports\": [
      {
        \"file\": \"filename\",
        \"missing\": \"export name\",
        \"expected_signature\": \"expected signature\"
      }
    ],
    \"changed_signatures\": [
      {
        \"file\": \"filename\",
        \"export\": \"export name\",
        \"expected\": \"expected signature\",
        \"actual\": \"actual signature\"
      }
    ],
    \"missing_dependencies\": [\"dependencies that should have been added\"],
    \"unexpected_dependencies\": [\"dependencies added that weren't planned\"]
  },
  \"recommendations\": [
    \"Specific actions to resolve differences\"
  ]
}
```

### 6. Determine Next Steps

#### If Exact Match or Acceptable Variations:
- Move task file to `tasks/completed/[TASK_ID].json`
- Update task status in task list
- Mark task as complete
- Proceed to next task

#### If Problematic Differences:
- Create detailed mismatch report
- Pause implementation workflow
- Prepare for `resolve_mismatch` command
- Do not mark task as complete

### 7. Update Project Status
- Update `codebase_manifest.json` with current state
- Update task status in task list
- Record completion timestamp
- Note any lessons learned

## Validation Criteria:

### Critical Checks:
- All planned exports are implemented
- Function signatures match expected
- All planned files are created
- No breaking changes to existing functionality
- All planned dependencies are added

### Quality Checks:
- Code compiles without errors
- Basic functionality works
- Integration points function correctly
- Error handling is appropriate

## Success Criteria:
- Validation completed successfully
- Differences categorized and documented
- Next steps clearly identified
- Project status updated
- Task completion status determined

## Output Summary:
Report validation results:
- Overall match status (MATCH/MINOR_DIFFERENCES/MAJOR_DIFFERENCES)
- Summary of what was implemented
- List of any differences found
- Recommendations for next steps
- Task completion status

Use `resolve_mismatch` command if major differences were found, otherwise the task is complete."
```