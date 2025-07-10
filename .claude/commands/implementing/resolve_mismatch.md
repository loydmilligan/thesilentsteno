# Resolve Mismatch Command

```bash
claude-code "Resolve discrepancies between expected and actual implementation in manifest-driven development.

## Task: Resolve Implementation Mismatch

**Task ID:** [TASK_ID - e.g., Task-1.1]

Start your response with: "🚨 **RESOLVE_MISMATCH EXECUTING** - Resolving [TASK_ID] discrepancies"

## Resolution Process:

### 1. Load Mismatch Context
- Read the comparison report from `tasks/validation/[TASK_ID]-comparison.json`
- Understand the specific differences found
- Load the original task context and expected manifest
- Review the current actual manifest

### 2. Analyze Differences

#### Categorize Each Difference:

**Critical Issues (Must Fix):**
- Missing core functionality
- Broken API contracts
- Missing essential exports
- Incomplete implementation of task requirements
- Breaking changes to existing functionality

**Implementation Variations (May Accept):**
- Different internal implementation approach
- Additional helper functions/classes
- Enhanced error handling
- Better validation than planned
- More comprehensive documentation

**Scope Creep (Evaluate):**
- Features beyond task scope
- Additional files created
- Unplanned dependencies
- Architectural changes beyond requirements

### 3. Present Resolution Options

For each significant difference, provide these options:

#### Option A: Fix Implementation
- Modify code to match expected manifest
- Pros: Maintains planned architecture, predictable results
- Cons: May remove beneficial improvements, rework required

#### Option B: Update Expected Manifest
- Accept the current implementation as correct
- Update the expected manifest to match actual
- Pros: Keeps beneficial changes, less rework
- Cons: May accept scope creep, deviates from plan

#### Option C: Split Into Separate Task
- Extract extra functionality into a new task
- Keep core implementation focused
- Pros: Maintains task scope, captures extra work
- Cons: Additional task management overhead

#### Option D: Hybrid Approach
- Keep beneficial changes, fix critical issues
- Update manifest for improvements, fix problems
- Pros: Best of both worlds
- Cons: More complex resolution

### 4. Interactive Resolution Process

For each difference, present:
```
DIFFERENCE: [Description of the difference]
SEVERITY: [Critical/Minor/Scope Creep]
IMPACT: [How this affects the project]

RESOLUTION OPTIONS:
A) Fix implementation to match expected
B) Update expected manifest to match actual  
C) Split into separate task
D) Hybrid approach

RECOMMENDATION: [Based on severity and impact]

What is your preferred resolution? [A/B/C/D]
```

### 5. Execute Resolution Actions

#### For \"Fix Implementation\" (A):
- Modify code to match expected manifest
- Remove/change problematic implementations
- Ensure all planned exports are correctly implemented
- Verify integration points work as expected
- Re-run implementation validation

#### For \"Update Expected Manifest\" (B):
- Update the expected manifest to match current implementation
- Document why changes were accepted
- Verify new manifest is internally consistent
- Update task completion criteria
- Mark task as complete with notes

#### For \"Split Into Separate Task\" (C):
- Extract extra functionality into new task definition
- Revert code to original task scope
- Create new task file for extracted functionality
- Update task list with new task
- Complete original task with focused scope

#### For \"Hybrid Approach\" (D):
- Keep beneficial improvements
- Fix critical issues
- Update manifest for accepted changes
- Document decision rationale
- Verify final state meets requirements

### 6. Document Resolution
Create resolution record in `tasks/resolutions/[TASK_ID]-resolution.json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"resolution_timestamp\": \"[timestamp]\",
  \"original_differences\": [\"list of differences found\"],
  \"resolutions_applied\": [
    {
      \"difference\": \"description of difference\",
      \"resolution_type\": \"fix_implementation|update_manifest|split_task|hybrid\",
      \"rationale\": \"why this resolution was chosen\",
      \"actions_taken\": [\"specific actions performed\"]
    }
  ],
  \"final_status\": \"[RESOLVED/PARTIAL/DEFERRED]\",
  \"lessons_learned\": [\"insights for future tasks\"],
  \"updated_manifest\": { \"final manifest after resolution\" }
}
```

### 7. Validation After Resolution
- Re-run manifest generation
- Verify all critical issues are resolved
- Ensure task requirements are met
- Check that no new issues were introduced
- Confirm integration points still work

### 8. Update Project State
- Update `codebase_manifest.json` with final state
- Move task to completed with resolution notes
- Update task list with lessons learned
- Document any process improvements identified

## Resolution Principles:

### Prioritization:
1. **Functionality First:** Ensure core requirements are met
2. **Maintain Quality:** Don't sacrifice quality for compliance
3. **Preserve Improvements:** Keep beneficial changes when possible
4. **Document Decisions:** Record rationale for future reference

### Decision Guidelines:
- **Critical functionality:** Always implement as planned
- **API contracts:** Maintain consistency unless improvement is clear
- **Internal implementation:** Allow flexibility for better solutions
- **Scope creep:** Evaluate benefit vs. complexity
- **Dependencies:** Minimize unnecessary additions

## Success Criteria:
- All critical differences resolved
- Final implementation meets task requirements
- Manifest accurately reflects actual state
- Resolution rationale documented
- Project ready to proceed to next task

## Output Summary:
Report resolution results:
- Number of differences resolved
- Resolution approaches used
- Final task status
- Any new tasks created
- Lessons learned for future tasks
- Project ready for next task

The mismatch resolution is complete and the task can now be marked as finished."
```