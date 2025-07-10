# Implement Task Command

```bash
claude-code "Implement a prepared task with full context using manifest-driven development.

## Task: Implement Prepared Task

**Task File:** [PATH_TO_TASK_FILE - e.g., tasks/prepared/Task-1.1.json]

Start your response with: "⚙️ **IMPLEMENT_TASK EXECUTING** - Implementing [TASK_FILE]"

## Implementation Process:

### 1. Load Task Context
- Read the complete task file with all context
- Understand current manifest state
- Review expected post-task manifest
- Study implementation notes and approach

### 2. Verify Prerequisites
- Check that all prerequisite tasks are completed
- Ensure current codebase state matches expected starting point
- Verify required dependencies are available

### 3. Implementation Phase

#### Create New Files:
For each file in `files_to_create`:
- Create file with proper directory structure
- Implement all planned exports with full functionality
- Add proper imports as specified in expected manifest
- Include comprehensive comments and documentation
- Follow coding standards and conventions

#### Modify Existing Files:
For each file in `files_to_modify`:
- Make specified changes while preserving existing functionality
- Update exports if needed
- Maintain backward compatibility where possible
- Update imports if dependencies change

#### Add Dependencies:
- Update package.json with new dependencies
- Install packages if needed
- Update configuration files (tsconfig.json, etc.)
- Verify compatibility with existing dependencies

### 4. Integration and Testing

#### Code Integration:
- Ensure all imports resolve correctly
- Verify function signatures match expected manifest
- Test integration points with existing code
- Handle error cases and edge conditions

#### Basic Testing:
- Verify code compiles without errors
- Test basic functionality of new features
- Ensure existing functionality still works
- Run any available test suites

### 5. Code Quality Checks
- Proper error handling
- Consistent coding style
- Adequate comments and documentation
- No console.log or debugging code left behind
- Proper TypeScript typing (if applicable)

### 6. Validation Against Expected Manifest
- Verify all planned exports are implemented
- Check that file purposes match implementation
- Ensure side effects are as expected
- Validate integration points work correctly

### 7. Documentation Updates
- Update README.md if new features are user-facing
- Add inline documentation for complex functions
- Update any API documentation
- Note any deviations from the original plan

### 8. Final Verification
- Code builds successfully
- All planned functionality is working
- No breaking changes to existing code
- Implementation matches acceptance criteria
- Ready for manifest validation

## Implementation Standards:

### Code Quality:
- Write clean, readable code
- Use meaningful variable and function names
- Add comments for complex logic
- Follow established patterns in the codebase
- Handle errors gracefully

### Testing:
- Test happy path scenarios
- Test error conditions
- Verify integration with existing code
- Ensure backward compatibility

### Documentation:
- Document public APIs
- Explain complex algorithms
- Note any assumptions or limitations
- Update relevant documentation files

## Success Criteria:
- All files created/modified as planned
- Code compiles and runs without errors
- Basic functionality testing passes
- Implementation matches acceptance criteria
- No breaking changes to existing functionality
- Code quality standards met

## Output Summary:
Report on implementation results:
- Files created and modified
- Dependencies added
- Key functionality implemented
- Any deviations from the plan
- Testing results
- Ready for manifest validation

The task implementation is complete and ready for validation using the `check_task` command."
```