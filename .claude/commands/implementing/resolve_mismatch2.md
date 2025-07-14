# Resolve Mismatch Command (Revised)

```bash
claude-code "Resolve discrepancies between expected and actual implementation in manifest-driven development.

Start your response with: '🚨 **RESOLVE_MISMATCH EXECUTING** - Resolving [TASK_ID] discrepancies'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ TASK_ID parameter provided (format: Task-X.X)
- ✅ tasks/validation/[TASK_ID]-validation.json exists
- ✅ Validation report shows MAJOR_ISSUES or CRITICAL_FAILURE
- ✅ tasks/prepared/[TASK_ID].json exists (original task context)
- ✅ Current codebase is accessible for analysis
- ✅ Differences are documented in validation report

**Input Validation Results:**
- [ ] TASK_ID: [TASK_ID] - [VALID/INVALID]
- [ ] Validation report: [EXISTS/MISSING] - tasks/validation/[TASK_ID]-validation.json
- [ ] Validation status: [MAJOR_ISSUES/CRITICAL_FAILURE/OTHER] - [NEEDS_RESOLUTION/NO_ISSUES]
- [ ] Task preparation: [EXISTS/MISSING] - tasks/prepared/[TASK_ID].json
- [ ] Codebase access: [ACCESSIBLE/INACCESSIBLE]
- [ ] Documented differences: [AVAILABLE/MISSING] in validation report

**❌ STOP EXECUTION if no major issues found or required inputs missing**

---

## Task: Resolve Implementation Mismatch

**Task ID:** [TASK_ID]

## Mismatch Resolution Process:

### 1. Load Mismatch Context
- Read validation report from `tasks/validation/[TASK_ID]-validation.json`
- Load original task context from `tasks/prepared/[TASK_ID].json`
- Extract baseline, expected, and actual manifests
- Review all documented differences by severity
- Understand the scope and impact of each mismatch

### 2. Analyze and Categorize Differences

#### 🚨 Critical Issues (Must Fix):
- Missing core functionality required for task completion
- Broken API contracts or existing functionality
- Missing essential exports or interfaces
- Incomplete implementation of acceptance criteria
- Breaking changes that affect existing code
- Security vulnerabilities introduced
- Build failures or compilation errors

#### ⚠️ Concerning Issues (Evaluate for Fix):
- Missing planned features (non-critical)
- Changed function signatures from expected
- Missing planned files or dependencies
- Scope creep beyond task requirements
- Performance regressions
- Inconsistent code quality or standards

#### ℹ️ Minor Issues (May Accept):
- Different internal implementation approach
- Additional helper functions or utilities
- Enhanced error handling beyond planned
- Better validation than originally planned
- More comprehensive documentation
- Code organization improvements

### 3. Present Resolution Options for Each Difference

For each significant difference, present structured options:

```
🔍 DIFFERENCE ANALYSIS
Type: [Missing Feature/Changed Signature/Scope Creep/etc.]
Severity: [Critical/Concerning/Minor]
Description: [Detailed description of the difference]
Impact: [How this affects the project and future tasks]
Location: [Specific files/functions affected]

💡 RESOLUTION OPTIONS:

Option A: Fix Implementation
- Modify code to match expected manifest exactly
- Pros: Maintains planned architecture, predictable results
- Cons: May remove beneficial improvements, requires rework
- Effort: [Low/Medium/High]

Option B: Update Expected Manifest  
- Accept current implementation as correct
- Update expected manifest to match actual state
- Pros: Keeps beneficial changes, less rework required
- Cons: May accept scope creep, deviates from plan
- Effort: [Low/Medium/High]

Option C: Split Into Separate Task
- Extract extra functionality into new future task
- Revert code to original task scope
- Pros: Maintains task focus, captures extra work for later
- Cons: Additional task management, feature delay
- Effort: [Low/Medium/High]

Option D: Hybrid Approach
- Keep beneficial changes, fix critical issues
- Update manifest for improvements, fix problems
- Pros: Best of both worlds, pragmatic solution
- Cons: More complex resolution, requires careful analysis
- Effort: [Low/Medium/High]

🎯 RECOMMENDATION: [A/B/C/D] - [Reasoning for recommendation]
```

### 4. Interactive Resolution Decision Process

**Note:** This process requires human input for each significant difference.

For each difference requiring resolution:
1. Present the analysis and options above
2. **REQUEST USER DECISION:** \"Which resolution approach do you prefer for this difference? [A/B/C/D]\"
3. Wait for user response before proceeding
4. Document the chosen resolution approach
5. Move to next difference

### 5. Execute Resolution Actions

#### For \"Fix Implementation\" (Option A):
- Identify specific code changes needed
- Modify implementation to match expected manifest
- Remove or change problematic code sections
- Ensure all planned exports are correctly implemented
- Verify integration points work as expected
- Re-test modified functionality

#### For \"Update Expected Manifest\" (Option B):
- Update the expected manifest in preparation file
- Document rationale for accepting current implementation
- Verify updated manifest is internally consistent  
- Update acceptance criteria if necessary
- Validate that changes don't break future task dependencies

#### For \"Split Into Separate Task\" (Option C):
- Identify functionality to extract into new task
- Create new task definition for extracted features
- Revert current code to original task scope
- Update task list with new task and dependencies
- Complete original task with focused scope

#### For \"Hybrid Approach\" (Option D):
- Keep beneficial improvements in current implementation
- Fix identified critical issues
- Update expected manifest for accepted improvements
- Document decision rationale for each change
- Ensure final state meets core task requirements

### 6. Document Resolution Decisions

Create comprehensive resolution record at `tasks/resolutions/[TASK_ID]-resolution.json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"resolution_timestamp\": \"[ISO timestamp]\",
  \"original_validation_status\": \"[MAJOR_ISSUES/CRITICAL_FAILURE]\",
  \"differences_found\": [
    {
      \"type\": \"[difference type]\",
      \"severity\": \"[Critical/Concerning/Minor]\",
      \"description\": \"[detailed description]\",
      \"resolution_chosen\": \"[A/B/C/D]\",
      \"resolution_rationale\": \"[why this resolution was chosen]\",
      \"actions_taken\": [\"specific actions performed\"],
      \"effort_required\": \"[Low/Medium/High]\"
    }
  ],
  \"new_tasks_created\": [
    {\"task_id\": \"[new task ID]\", \"description\": \"[what was split off]\"}
  ],
  \"final_status\": \"[RESOLVED/PARTIALLY_RESOLVED/REQUIRES_REIMPLEMENTATION]\",
  \"lessons_learned\": [\"insights for improving future tasks\"],
  \"process_improvements\": [\"suggestions for better workflow\"],
  \"updated_expected_manifest\": {\"final expected manifest after resolution\"},
  \"ready_for_validation\": \"[YES/NO]\"
}
```

### 7. Post-Resolution Validation

After applying resolutions:
- Re-run manifest generation to capture current state
- Verify all critical issues have been addressed
- Ensure task requirements are still met
- Check that no new issues were introduced
- Confirm integration points still function correctly
- Validate that build/compile still works

### 8. Update Project State

- Update task preparation file with any manifest changes
- Create new task definitions for any split-off work
- Document lessons learned in project notes
- Update task dependencies if changes affect future work
- Prepare for re-validation or commit as appropriate

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ All differences analyzed and resolution decisions made
- ✅ Resolution actions executed successfully  
- ✅ Resolution record created with full documentation
- ✅ Any new tasks created and documented
- ✅ Updated manifests reflect final state
- ✅ Post-resolution validation completed
- ✅ Clear status for next steps provided

**Output Validation Results:**
- [ ] Difference analysis: [COMPLETED/INCOMPLETE] - [COUNT] differences processed
- [ ] Resolution execution: [SUCCESSFUL/PARTIAL/FAILED] 
- [ ] Resolution record: [CREATED/FAILED] - tasks/resolutions/[TASK_ID]-resolution.json
- [ ] New tasks: [CREATED/NONE] - [COUNT] new tasks if any
- [ ] Manifest updates: [COMPLETED/FAILED]
- [ ] Post-resolution validation: [PASSED/FAILED]
- [ ] Next steps: [CLEAR/UNCLEAR]

**✅ SUCCESS CRITERIA MET** - Mismatch resolved, ready for next action
**❌ FAILURE** - Resolution incomplete, may require manual intervention

## Resolution Summary Report:
- Task ID: [TASK_ID]
- Total Differences Processed: [COUNT]
- Critical Issues Resolved: [COUNT]
- Concerning Issues Resolved: [COUNT] 
- Minor Issues Accepted: [COUNT]
- New Tasks Created: [COUNT] - [list if any]
- Resolution Approach Used: [Primary approach used]
- Final Status: [RESOLVED/PARTIALLY_RESOLVED/REQUIRES_REIMPLEMENTATION]
- Resolution Record: tasks/resolutions/[TASK_ID]-resolution.json
- Ready for Next Action: [CHECK_TASK/COMMIT_TASK/MANUAL_REVIEW]

## Recommended Next Steps:
Based on resolution outcomes:
- If RESOLVED: Re-run check_task to validate fixes
- If PARTIALLY_RESOLVED: Continue resolution process
- If REQUIRES_REIMPLEMENTATION: Re-run implement_task with updated context
- If new tasks created: Update task list and prioritize

The mismatch resolution process is complete."
```