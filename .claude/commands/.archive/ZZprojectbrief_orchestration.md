# PRODUCTBRIEF ORCHESTRATION COMMAND

Think deeply about this comprehensive project validation challenge. You are about to orchestrate multiple specialized agents to conduct thorough project analysis, feature development, and validated MVP creation through systematic research and testing.

**Variables:**

project_concept: $ARGUMENTS
domain_context: $ARGUMENTS
output_dir: $ARGUMENTS

**ARGUMENTS PARSING:**
Parse the following arguments from "$ARGUMENTS":
1. `project_concept` - Basic project idea (1 sentence) and brief description including target users, main problem, and key concepts
2. `domain_context` - Technical domain or industry context (e.g., 'electronics', 'web development', 'data analysis')
3. `output_dir` - Directory where ProductBrief.json and validation reports will be saved

**PHASE 1: PROJECT CONCEPT ANALYSIS & PRODUCTBRIEF INITIALIZATION**
Deeply analyze the project concept to extract and structure core components:
- Parse user's problem statement and target user assumptions
- Identify key concepts and strategic objectives
- Extract implied technical requirements and constraints
- Determine educational, accessibility, or other special considerations
- Assess domain-specific challenges and opportunities

Create initial ProductBrief.json structure with:
```json
{
  "project_core": {
    "title": "[Extracted from concept]",
    "description": "[Structured from user input]",
    "problem_statement": "[Core problem being solved]",
    "strategic_objectives": "[Business/project goals]",
    "domain": "[Technical domain]",
    "special_considerations": ["educational", "accessibility", etc.]
  },
  "research_intelligence": {
    "user_research": {},
    "market_research": {},
    "validation_history": []
  },
  "feature_universe": {
    "all_features": [],
    "feasibility_scores": {},
    "value_scores": {},
    "mvp_test_results": []
  },
  "mvp_evolution": {
    "current_mvp": {},
    "validation_scores": {},
    "refinement_history": []
  }
}
```

**PHASE 2: PARALLEL RESEARCH ORCHESTRATION**
Deploy market and user research agents simultaneously with focused scope:

**Market Research Deployment (5-8 Agents):**
Deploy market research agents using adapted market research orchestration:
- **Competitive Landscape Analyst** - Direct and indirect competitors
- **Market Size & Opportunity** - TAM, SAM, SOM analysis  
- **Industry Trends Researcher** - Technology and business trends
- **Regulatory & Risk Assessment** - Compliance and market barriers
- **Customer Segmentation Specialist** - Market demographics and segments
- **Pricing & Business Model Analyst** - Revenue strategies and models
- **Geographic Market Analyst** (if 7+ agents) - Regional opportunities
- **Partnership & Ecosystem Mapper** (if 8+ agents) - Strategic landscape

**User Research Deployment (5-8 Agents):**
Deploy user research agents using adapted user research orchestration:
- **Primary Persona Architect** - Core user personas and characteristics
- **User Journey Mapper** - End-to-end user experience and touchpoints
- **Needs & Pain Points Analyst** - User problems and unmet needs
- **Behavioral Patterns Researcher** - User habits and decision-making
- **Demographic Segmentation Specialist** - Age, location, income patterns
- **Psychographic Profile Developer** - Values, attitudes, lifestyle factors
- **Technology Adoption Researcher** (if 7+ agents) - Digital behavior patterns
- **Accessibility & Inclusion Specialist** (if 8+ agents) - Diverse user needs

**Research Agent Task Modifications:**
Each agent receives ProductBrief project_core context and is instructed to:
- Focus research specifically on validating/refuting user's initial assumptions
- Emphasize findings that directly impact the stated problem and strategic objectives
- Balance fidelity to user vision with discovery of new opportunities
- Provide confidence levels for all major findings

**Research Output Integration:**
Update ProductBrief.json with:
- `research_intelligence.market_research` - Comprehensive market findings
- `research_intelligence.user_research` - Detailed user intelligence  
- `validation_history` - Record of initial assumption validation

**PHASE 3: FEATURE GENERATION ORCHESTRATION**
Deploy comprehensive feature generation using features.md orchestration (5-8 agents):

**Feature Generation Deployment:**
- **Core Functionality Specialist** - Essential features for MVP
- **User Experience Innovation** - Advanced UX/UI features
- **Technical Infrastructure** - Backend, security, performance features
- **Business Value Creator** - Monetization, analytics, growth features
- **Future Vision Architect** - Cutting-edge, forward-looking features
- **Accessibility Champion** (if 6+ agents) - Inclusive design features
- **Mobile-First Designer** (if 7+ agents) - Mobile-specific capabilities
- **Integration Specialist** (if 8+ agents) - Third-party and API features

**Feature Generation Context:**
Provide complete ProductBrief context including research intelligence to ensure features are:
- Grounded in validated market and user research
- Aligned with strategic objectives and problem statement
- Appropriate for identified domain and technical constraints

**Feature Output Integration:**
Update ProductBrief.json with:
- `feature_universe.all_features` - Comprehensive feature catalog with detailed specifications

**PHASE 4: FEATURE EVALUATION & DECISION MATRIX**
Deploy specialized evaluation agents for systematic feature assessment:

**Scope Agent Deployment:**
```
TASK: Technical Complexity Assessment for ProductBrief Features

Evaluate each feature for implementation complexity considering:
- Technical difficulty and development time
- Resource requirements (team size, skills needed)
- Infrastructure and technology stack demands
- Testing and quality assurance complexity
- Integration challenges with other features
- Maintenance and support overhead

SCORING SYSTEM: 1-10 scale
- 1-3: Low Complexity (weeks of development)
- 4-6: Medium Complexity (1-3 months of development)  
- 7-8: High Complexity (3-6 months of development)
- 9-10: Very High Complexity (6+ months or significant technical risk)

OUTPUT: Complexity score with brief justification for each feature
```

**Value Proposition Agent Deployment:**
```
TASK: Strategic Value Assessment for ProductBrief Features

Evaluate each feature for business and strategic value considering:
- Strategic alignment with core project objectives
- Direct contribution to solving stated problem
- Business impact potential (revenue, user acquisition, retention)
- Competitive advantage and differentiation value
- User value delivery against research findings
- Synergy potential with other features

SCORING SYSTEM: 1-10 scale
- 1-3: Low Value (nice-to-have, minimal impact)
- 4-6: Medium Value (solid contribution to objectives)
- 7-8: High Value (significant strategic impact)
- 9-10: Critical Value (essential for project success)

OUTPUT: Value score with strategic justification for each feature
```

**Orchestrator Decision Matrix:**
For each feature, apply decision logic:

```
IF Value >= 7 AND Complexity <= 4: 
    DECISION = "Accept for MVP Testing"
    
IF Value >= 7 AND Complexity >= 5:
    DECISION = "Accept with Complexity Flag"
    
IF Value <= 4 (regardless of complexity):
    DECISION = "Reject Feature"
    
IF Value = 5-6 AND Complexity >= 6:
    DECISION = "Send for Refinement: Reduce Complexity"
    
IF Value = 5-6 AND Complexity <= 5:
    DECISION = "Send for Refinement: Increase Value Alignment"
```

**Refinement Process:**
For features requiring refinement, send back to feature generation with specific instructions:
- **Complexity Reduction**: "Simplify implementation while maintaining core user value"
- **Value Enhancement**: "Increase strategic alignment and business impact"
- **Feature Decomposition**: "Break into smaller, simpler components"

Limit to 1 refinement round per feature before elimination.

**Feature Pool Validation:**
After decision matrix processing, validate feature pool size:
- If < 8 features accepted: Consult user for guidance on minimum feature requirements
- If 8-15 features: Proceed to MVP testing
- If > 15 features: Consider raising complexity/value thresholds

**Update ProductBrief:**
- `feature_universe.feasibility_scores` - Complexity assessments
- `feature_universe.value_scores` - Strategic value assessments
- `refinement_history` - Record of refinement actions

**PHASE 5: MVP TESTING ORCHESTRATION**
Conduct systematic MVP combination testing using validated features:

**MVP Testing Protocol:**
1. **Create Feature Pool**: All features that passed decision matrix
2. **Generate MVP Combinations**: Create 8-12 different MVP feature sets (3-5 features each)
3. **Market Agent Testing**: Deploy market research agents to evaluate each MVP combination
4. **User Agent Testing**: Deploy user research agents to evaluate each MVP combination

**MVP Testing Agent Instructions:**
```
TASK: MVP Combination Validation

You will evaluate multiple MVP combinations (3-5 features each) against your research domain.

For each MVP combination:
1. Assume the app works perfectly with only these features
2. Score how well this MVP serves your research findings:
   - Market fit and competitive positioning (Market Agents)
   - User needs satisfaction and adoption likelihood (User Agents)
3. Provide score (1-10) and brief justification
4. Identify any critical gaps or concerns

TEST 8 DIFFERENT MVP COMBINATIONS
Track which features appear in highest-scoring MVPs
```

**MVP Testing Analysis:**
1. **Feature Frequency Analysis**: Which features appear most in high-scoring MVPs?
2. **Score Distribution**: Which MVP combinations consistently score highest?
3. **Consensus Identification**: Features chosen by both market and user agents
4. **Gap Analysis**: Critical needs not addressed by any MVP combination

**Update ProductBrief:**
- `feature_universe.mvp_test_results` - Complete testing data and analysis

**PHASE 6: MVP FINALIZATION & VALIDATION**
Select optimal MVP based on testing data and conduct final validation:

**MVP Selection Algorithm:**
1. **Identify Top Features**: Features appearing in highest-scoring MVPs (score >= 7)
2. **Consensus Weighting**: Prioritize features chosen by both market and user agents  
3. **Strategic Alignment Check**: Ensure MVP addresses core strategic objectives
4. **Complexity Validation**: Verify total complexity is reasonable for available resources

**Final MVP Validation Criteria:**
- Average feature value score >= 6.5
- Total MVP complexity score <= 25 (for 5 features)
- Strategic objectives coverage >= 80%
- Market fit score >= 7
- User satisfaction score >= 7

**Final Validation Process:**
If MVP passes validation criteria:
- Finalize ProductBrief.json with complete MVP specification
- Generate executive summary with validation audit trail

If MVP fails validation:
- Identify specific failure points
- Recommend refinement actions or feature substitutions
- Return to Phase 4 for targeted improvements

**Update ProductBrief:**
- `mvp_evolution.current_mvp` - Final MVP specification
- `mvp_evolution.validation_scores` - Complete validation results

**EXECUTION PRINCIPLES:**

**Systematic Validation:**
- Every decision point has clear criteria and audit trail
- Multiple validation gates ensure quality at each phase
- Research findings drive all feature and MVP decisions
- Balance user vision with data-driven optimization

**Iterative Refinement:**
- Features improve through targeted feedback loops
- MVP selection based on empirical testing, not assumptions
- Continuous validation against strategic objectives
- Practical complexity constraints guide all decisions

**Strategic Alignment:**
- Maintain focus on core problem and strategic objectives
- Balance innovation with implementation feasibility
- Ensure MVP delivers meaningful user and business value
- Create competitive advantage through validated differentiation

**Quality Assurance:**
- Comprehensive research foundation before feature generation
- Multi-dimensional feature evaluation (value, complexity, market fit, user needs)
- Systematic MVP testing with quantitative validation
- Complete documentation and reasoning for all decisions

**ULTRA-THINKING DIRECTIVE:**
Before beginning deployment, engage in extended analysis about:

**Project Vision & Research Strategy:**
- How to structure research to validate user assumptions while discovering new opportunities
- Optimal balance between fidelity to user vision and research-driven insights
- Methods for ensuring research quality and actionable findings
- Strategies for synthesizing market and user intelligence into coherent project understanding

**Feature Development & Evaluation Strategy:**
- How to generate features that are both innovative and grounded in research
- Optimal decision matrix criteria for balancing value and complexity
- Refinement strategies that improve features without compromising core value
- Methods for ensuring feature alignment with strategic objectives

**MVP Testing & Validation Methodology:**
- Systematic approaches for testing multiple MVP combinations against research findings
- Methods for identifying optimal feature combinations through empirical testing
- Validation criteria that ensure MVP viability across multiple dimensions
- Strategies for handling conflicting validation results and edge cases

**Process Optimization & Quality Control:**
- Coordination strategies for managing multiple agent deployments and feedback loops
- Quality assurance methods for ensuring consistent results across all phases
- Documentation and audit trail requirements for complete decision transparency
- Error handling and recovery strategies for managing complex orchestration workflows

Begin execution with deep analysis of the project concept and proceed systematically through research, feature development, evaluation, and MVP validation, leveraging multiple expert perspectives for comprehensive project planning and validated product development.