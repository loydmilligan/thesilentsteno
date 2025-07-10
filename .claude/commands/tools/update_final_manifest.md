# Update Final Manifest Command

```bash
claude-code "Update the proposed final manifest based on implementation learnings and project evolution.

## Task: Update Proposed Final Manifest

**Trigger:** [After major milestone/phase completion, significant architectural changes, or periodic review]

Start your response with: "📝 **UPDATE_FINAL_MANIFEST EXECUTING** - Updating proposed final manifest"

## Update Process:

### 1. Analyze Implementation Learnings
- Review completed tasks and their resolutions
- Identify architectural changes discovered during implementation
- Note new requirements or features that emerged
- Assess changes in dependencies or integration points

### 2. Compare Current vs. Proposed
- Load current `codebase_manifest.json` (actual state)
- Load `docs/proposed_final_manifest.json` (target state)
- Identify significant differences between actual and proposed

### 3. Evaluate Architectural Evolution
- **Better Patterns Discovered:** New approaches found during implementation
- **Scope Changes:** Features added/removed from original plan
- **Integration Changes:** External systems or APIs changed
- **Performance Optimizations:** Architecture improvements identified
- **Security Considerations:** New security requirements discovered

### 4. Update Proposed Final Manifest
Revise the proposed final manifest to reflect:
- **New/Modified Components:** Based on implementation learnings
- **Updated API Signatures:** Better parameter types or return values discovered
- **Enhanced Architecture:** Improved data flow or system design
- **Additional Dependencies:** New tools or libraries proven beneficial
- **Refined Integration Points:** Better external system interactions

### 5. Document Changes
Create update log in `docs/manifest_evolution.md`:
```markdown
## Manifest Update [Date]

### Trigger
[What prompted this update - milestone completion, architectural discovery, etc.]

### Key Changes
- [Change 1]: [Rationale]
- [Change 2]: [Rationale]
- [Change 3]: [Rationale]

### Impact Assessment
- **Existing Tasks:** [How this affects remaining tasks]
- **Architecture:** [How this changes system design]
- **Dependencies:** [New/changed dependencies]
- **Timeline:** [Impact on project timeline]

### Lessons Learned
- [Learning 1]
- [Learning 2]
- [Learning 3]
```

### 6. Update Related Documentation
- Update `docs/architecture.md` if system design changed
- Update `docs/api.md` if API signatures changed
- Update `tasks/task_list.md` if remaining tasks need modification
- Update `README.md` if project scope or features changed

### 7. Validate Updated Manifest
- Ensure internal consistency
- Verify all dependencies are accounted for
- Check that integration points are realistic
- Validate that remaining tasks align with updated manifest

### 8. Commit Changes
```bash
git add docs/proposed_final_manifest.json docs/manifest_evolution.md
git commit -m \"Update proposed final manifest: [brief description of changes]\"
```

## Update Triggers:

### Major Milestones:
- Completion of Phase 1 (Project Setup)
- Completion of Phase 2 (Core System Functionality)
- Completion of Phase 3 (Feature Implementation)
- Completion of Phase 4 (Integration and Polish)

### Significant Discoveries:
- Better architectural patterns found
- New integration requirements discovered
- Performance or security improvements identified
- Major scope changes required

### Periodic Reviews:
- Every 5-10 completed tasks
- Monthly reviews for long projects
- Before major feature implementations

## Success Criteria:
- Proposed final manifest reflects current architectural understanding
- Changes are well-documented with rationale
- Related documentation is updated
- Remaining tasks align with updated manifest
- Evolution history is preserved

This ensures the proposed final manifest evolves with the project and captures implementation learnings."
```