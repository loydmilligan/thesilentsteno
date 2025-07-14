# Check Task Command (Revised)

```bash
claude-code "Verify task implementation matches expected changes using three-way manifest comparison.

Start your response with: '✅ **CHECK_TASK EXECUTING** - Validating [TASK_ID] implementation'

## 📋 REQUIRED INPUTS CHECK

Before proceeding, verify these inputs exist and are valid:

🔍 **INPUT REQUIREMENTS:**
- ✅ TASK_ID parameter provided (format: Task-X.X)
- ✅ tasks/prepared/[TASK_ID].json exists (task preparation file)
- ✅ Preparation file contains baseline_manifest
- ✅ Preparation file contains expected_manifest
- ✅ Current codebase exists and is accessible for analysis
- ✅ Implementation has been completed (implement_task was run)

**Input Validation Results:**
- [ ] TASK_ID: [TASK_ID] - [VALID/INVALID]
- [ ] Preparation file: [EXISTS/MISSING] - tasks/prepared/[TASK_ID].json
- [ ] Baseline manifest: [LOADED/MISSING] from preparation file
- [ ] Expected manifest: [LOADED/MISSING] from preparation file  
- [ ] Codebase access: [ACCESSIBLE/INACCESSIBLE]
- [ ] Implementation status: [COMPLETED/PENDING]

**❌ STOP EXECUTION if any required inputs are missing or invalid**

---

## Task: Validate Task Implementation

**Task ID:** [TASK_ID]

## Three-Way Validation Process:

### 1. Load Manifests for Comparison
- **Baseline Manifest:** Load from preparation file (the starting state)
- **Expected Manifest:** Load from preparation file (the planned end state)
- **Actual Manifest:** Generate fresh from current codebase (the real end state)

**Critical:** The baseline manifest MUST come from the preparation file, NOT from current codebase analysis.

### 2. Generate Actual Manifest from Current Codebase
Scan the current codebase to create a fresh manifest that represents the actual state:
- Analyze all source files for exports, functions, classes, interfaces
- Extract dependency information from package.json and imports
- Document current architecture and integration points
- Capture any side effects or configuration changes
- Note file structure and organization

### 3. Three-Way Manifest Comparison

#### Comparison A: Baseline → Expected (Plan Validation)
Verify the expected manifest was logically derived from baseline:
- **New Files:** Are planned additions reasonable and necessary?
- **File Changes:** Are planned modifications logical and minimal?
- **Dependencies:** Are new dependencies justified and minimal?
- **Architecture:** Are architectural changes well-reasoned?

#### Comparison B: Expected → Actual (Implementation Validation)  
Verify implementation followed the plan:
- **File-Level Comparison:**
  - All planned files created? [YES/NO - list missing]
  - All planned modifications made? [YES/NO - list missing]
  - Any unplanned files created? [YES/NO - list unexpected]
  
- **Export-Level Comparison:**
  - Functions: Signatures, parameters, return types match? [YES/NO]
  - Classes: Constructor, methods, properties match? [YES/NO]
  - Constants: Names, types, values match? [YES/NO]
  - Interfaces: Structure and properties match? [YES/NO]
  
- **Dependency Comparison:**
  - All planned dependencies added? [YES/NO - list missing]
  - Any unplanned dependencies added? [YES/NO - list unexpected]
  - Version compatibility maintained? [YES/NO]

#### Comparison C: Baseline → Actual (Change Validation)
Verify no unintended side effects or regressions:
- **Regression Check:** Any existing functionality broken? [YES/NO]
- **Scope Creep:** Changes beyond task requirements? [YES/NO]
- **Side Effects:** Unexpected architectural changes? [YES/NO]
- **Breaking Changes:** Any backward compatibility issues? [YES/NO]

### 4. Categorize All Differences

#### ✅ Acceptable Differences:
- Minor implementation details that don't affect public API
- Enhanced error handling beyond what was planned
- Additional helpful comments or documentation
- More comprehensive input validation
- Performance improvements that don't change interface

#### ⚠️ Concerning Differences:
- **Missing Planned Features:** Core functionality not implemented
- **Changed Signatures:** Function/method signatures differ from expected
- **Missing Files:** Planned files not created
- **Scope Creep:** Features beyond task requirements
- **Unplanned Dependencies:** New dependencies not in plan

#### 🚨 Critical Differences:
- **Broken Functionality:** Existing features no longer work
- **Breaking Changes:** API changes that break existing code
- **Security Issues:** New vulnerabilities introduced
- **Build Failures:** Code doesn't compile or run
- **Missing Requirements:** Acceptance criteria not met

### 5. Generate Detailed Comparison Report

Create comprehensive validation report as `tasks/validation/[TASK_ID]-validation.json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"validation_timestamp\": \"[ISO timestamp]\",
  \"overall_status\": \"[PASS/MINOR_ISSUES/MAJOR_ISSUES/CRITICAL_FAILURE]\",
  \"manifests\": {
    \"baseline_source\": \"tasks/prepared/[TASK_ID].json\",
    \"expected_source\": \"tasks/prepared/[TASK_ID].json\", 
    \"actual_source\": \"generated from current codebase\"
  },
  \"summary\": {
    \"files_created\": [\"list of files actually created\"],
    \"files_modified\": [\"list of files actually modified\"],
    \"exports_added\": [\"list of new exports found\"],
    \"dependencies_added\": [\"list of new dependencies found\"],
    \"plan_adherence\": \"[percentage or qualitative assessment]\"
  },
  \"differences\": {
    \"acceptable\": [
      {\"type\": \"enhancement\", \"description\": \"...\", \"impact\": \"positive\"}
    ],
    \"concerning\": [
      {\"type\": \"missing_feature\", \"description\": \"...\", \"impact\": \"moderate\"}
    ],
    \"critical\": [
      {\"type\": \"regression\", \"description\": \"...\", \"impact\": \"severe\"}
    ]
  },
  \"recommendations\": [
    \"Specific actions to address issues found\"
  ],
  \"next_action\": \"[PROCEED_TO_COMMIT/REQUIRE_FIXES/RESOLVE_MISMATCH]\"
}
```

### 6. Determine Validation Outcome

#### ✅ VALIDATION PASSED
- No critical or concerning differences found
- All acceptance criteria met
- Implementation matches expected manifest
- **Action:** Proceed to commit_task

#### ⚠️ MINOR ISSUES FOUND  
- Some concerning differences but no critical failures
- Core functionality complete but with deviations
- **Action:** Review differences, likely proceed to commit_task with notes

#### 🚨 MAJOR ISSUES FOUND
- Critical differences or missing core functionality
- Acceptance criteria not met
- Significant deviations from plan
- **Action:** Use resolve_mismatch command before proceeding

---

## 📤 REQUIRED OUTPUTS VERIFICATION

Verify these outputs were created successfully:

🎯 **OUTPUT REQUIREMENTS:**
- ✅ Three manifests successfully loaded/generated
- ✅ Three-way comparison completed
- ✅ All differences categorized by severity
- ✅ Validation report created
- ✅ Clear recommendation provided
- ✅ Overall validation status determined

**Output Validation Results:**
- [ ] Baseline manifest: [LOADED/FAILED] from preparation file
- [ ] Expected manifest: [LOADED/FAILED] from preparation file
- [ ] Actual manifest: [GENERATED/FAILED] from codebase
- [ ] Three-way comparison: [COMPLETED/INCOMPLETE]
- [ ] Difference categorization: [COMPLETE/INCOMPLETE]
- [ ] Validation report: [CREATED/FAILED] - tasks/validation/[TASK_ID]-validation.json
- [ ] Recommendation: [CLEAR/UNCLEAR]

**✅ SUCCESS CRITERIA MET** - Validation complete, recommendation provided
**❌ FAILURE** - Validation incomplete, cannot determine status

## Validation Summary Report:
- Task ID: [TASK_ID]
- Overall Status: [PASS/MINOR_ISSUES/MAJOR_ISSUES/CRITICAL_FAILURE]
- Files created vs planned: [X/Y created]
- Files modified vs planned: [X/Y modified]
- Dependencies added vs planned: [X/Y added]
- Critical differences: [COUNT] 
- Concerning differences: [COUNT]
- Acceptable differences: [COUNT]
- Validation report location: tasks/validation/[TASK_ID]-validation.json
- Recommended next action: [PROCEED_TO_COMMIT/REQUIRE_FIXES/RESOLVE_MISMATCH]

## Next Steps:
Based on validation results, proceed with the recommended action:
- If PASS/MINOR_ISSUES: Use commit_task command
- If MAJOR_ISSUES: Use resolve_mismatch command first"
```