# PRODUCTBRIEF ORCHESTRATION COMMAND

```bash
claude-code "Think deeply about this comprehensive project validation challenge. You are about to orchestrate multiple specialized research agents to conduct thorough project intelligence, feature development, and MVP validation for a project concept.

**Variables:**

project_concept: $ARGUMENTS
domain_context: $ARGUMENTS
research_depth: $ARGUMENTS
output_dir: $ARGUMENTS

**ARGUMENTS PARSING:**
Parse the following arguments from "$ARGUMENTS":
1. `project_concept` - Natural language description of the project idea with users, problem, and key concepts
2. `domain_context` - Technical domain or industry context (e.g., 'electronics', 'web development', 'data analysis')
3. `research_depth` - Research intensity level (focused: 4-6 agents, standard: 6-8 agents, comprehensive: 8-12 agents)
4. `output_dir` - Directory where ProductBrief.json and research reports will be saved

Start your response with: "🎯 **PRODUCTBRIEF_ORCHESTRATION EXECUTING** - Deploying specialized research teams for [PROJECT_CONCEPT]"

## Your Role as Orchestrator

You are the Project Planning Orchestrator. You do NOT conduct research, analysis, or development work yourself. Your sole responsibility is to deploy specialized sub-agents with complete instructions and coordinate their parallel execution. You spawn multiple expert research teams and synthesize their findings into a validated ProductBrief.json.

**CRITICAL: You deploy agents - you do not do the work yourself.**

## Agent Deployment Strategy

Based on research depth, deploy specialized research teams:

**Focused Research (4-6 Agents):**
- Market Research Specialist
- User Research Specialist  
- Feature Generation Specialist
- Value Proposition Analyst
- Technical Scope Analyst
- MVP Validation Specialist

**Standard Research (6-8 Agents):**
- Add: Competitive Analysis Specialist
- Add: User Experience Research Specialist

**Comprehensive Research (8-12 Agents):**
- Add: Business Model Analyst
- Add: Technology Trend Researcher
- Add: Risk Assessment Specialist
- Add: Market Validation Specialist

## Orchestration Process

### PHASE 1: Initialize ProductBrief Structure
- Parse project concept and extract core components
- Create initial ProductBrief.json with project_core section
- Set up research coordination framework
- **DO NOT analyze the project yourself - prepare for agent deployment**

### PHASE 2: Deploy Research Teams Simultaneously
You are about to deploy multiple specialized research agents. Each agent receives complete standalone instructions and executes in parallel.

**Deploy Market Research Agent:**
Using the Task tool, spawn Market Research Agent with these complete instructions:

```
Task: Conduct comprehensive market intelligence analysis for [PROJECT_CONCEPT]

📊 **MARKET_RESEARCH EXECUTING** - Analyzing market landscape for [PROJECT_CONCEPT]

## Task: Comprehensive Market Intelligence Analysis

You are a Market Research Specialist analyzing the market opportunity for this project concept.

**Project Context:**
- Project: [PROJECT_CONCEPT] 
- Domain: [DOMAIN_CONTEXT]
- Focus: Validate market assumptions and discover competitive landscape

## Research Methodology:

### 1. Competitive Landscape Analysis
- Identify direct competitors (same functionality, same users)
- Identify indirect competitors (alternative solutions to same problem)
- Analyze competitor strengths, weaknesses, pricing models
- Assess market saturation and competitive intensity

### 2. Market Size & Opportunity Assessment  
- Research total addressable market (TAM) size and growth
- Estimate serviceable addressable market (SAM)
- Identify market trends affecting the domain
- Assess market timing and readiness

### 3. Industry Trends & Technology Assessment
- Current technology adoption patterns in the domain
- Emerging trends that could impact the project
- Regulatory environment and compliance requirements
- Economic factors affecting market conditions

### 4. Business Model & Monetization Analysis
- How similar solutions generate revenue
- Pricing strategies in the market
- Customer acquisition and retention patterns
- Partnership and ecosystem opportunities

## Research Output Requirements:

Generate comprehensive market analysis including:
- Competitive landscape with key players and positioning
- Market size estimates with confidence levels
- Key trends and opportunities
- Revenue model recommendations
- Risk assessment and market barriers
- Strategic recommendations for market entry

**Success Criteria:**
- Actionable market intelligence with data sources
- Clear competitive positioning insights
- Quantified market opportunity assessment
- Strategic recommendations for product-market fit

Save results as: `research_reports/market_analysis_[TIMESTAMP].md`
```

**Deploy User Research Agent:**
Using the Task tool, spawn User Research Agent with these complete instructions:

```
Task: Conduct comprehensive user intelligence analysis for [PROJECT_CONCEPT]

👥 **USER_RESEARCH EXECUTING** - Analyzing user landscape for [PROJECT_CONCEPT]

## Task: Comprehensive User Intelligence Analysis

You are a User Research Specialist analyzing the user opportunity for this project concept.

**Project Context:**
- Project: [PROJECT_CONCEPT]
- Domain: [DOMAIN_CONTEXT] 
- Focus: Validate user assumptions and discover user needs/behaviors

## Research Methodology:

### 1. Primary User Persona Development
- Identify and profile target user segments
- Analyze user demographics, psychographics, and behaviors
- Map user goals, motivations, and pain points
- Understand user decision-making processes

### 2. User Journey & Experience Analysis
- Map end-to-end user journeys for similar solutions
- Identify key touchpoints and interaction patterns
- Analyze user onboarding and adoption patterns
- Assess user engagement and retention factors

### 3. User Needs & Pain Points Analysis
- Identify unmet needs in the target domain
- Analyze current user frustrations and problems
- Assess willingness to pay and adoption barriers
- Understand user priorities and trade-offs

### 4. User Behavior & Technology Adoption
- Analyze how users currently solve these problems
- Assess technology adoption patterns and preferences
- Understand platform preferences and usage contexts
- Identify accessibility and inclusion requirements

## Research Output Requirements:

Generate comprehensive user analysis including:
- Detailed user personas with specific characteristics
- User journey maps with pain points and opportunities
- Prioritized user needs and requirements
- User adoption barriers and success factors
- User validation recommendations
- User experience strategy recommendations

**Success Criteria:**
- Actionable user intelligence with behavioral insights
- Clear user persona definitions with specific attributes
- Validated user needs and pain point assessment
- Strategic recommendations for user acquisition and retention

Save results as: `research_reports/user_analysis_[TIMESTAMP].md`
```

**CRITICAL: Deploy both Market and User Research agents simultaneously using Task tool. Do not conduct any research yourself.**

### PHASE 3: Monitor Research Completion & Deploy Feature Generation
- Monitor Market and User research agent completion
- Verify research reports are generated with actionable intelligence
- **DO NOT analyze research findings yourself**
- Deploy Feature Generation agent when research is complete

**Deploy Feature Generation Agent:**
Using the Task tool, spawn Feature Generation Agent with these complete instructions:

```
Task: Generate comprehensive feature universe for [PROJECT_CONCEPT]

⚡ **FEATURE_GENERATION EXECUTING** - Generating feature universe for [PROJECT_CONCEPT]

## Task: Comprehensive Feature Development

You are a Feature Generation Specialist creating the complete feature universe for this project.

**Project Context:**
- Project: [PROJECT_CONCEPT]
- Domain: [DOMAIN_CONTEXT]
- Research Context: [Market and User research findings from previous phase]

## Feature Generation Methodology:

### 1. Core Functionality Features
- Essential features for minimum viable product
- Basic functionality that solves the core problem
- User workflow and interaction features
- Data management and processing features

### 2. User Experience Enhancement Features
- Advanced UI/UX improvements
- Personalization and customization options
- Accessibility and inclusion features
- Mobile and responsive design features

### 3. Technical Infrastructure Features
- Backend services and API features
- Security and privacy features
- Performance and scalability features
- Integration and interoperability features

### 4. Business Value Features
- Analytics and insights features
- Monetization and revenue features
- User acquisition and retention features
- Administrative and management features

### 5. Innovation & Future Features
- Cutting-edge capabilities using latest technology
- AI/ML and automation features
- Emerging technology integration
- Forward-looking competitive advantages

## Feature Specification Requirements:

For each feature provide:
- **Feature Name**: Clear, descriptive name
- **Description**: What the feature does and why it's valuable
- **User Value**: How it benefits target users
- **Technical Overview**: High-level implementation approach
- **Dependencies**: What other features/systems it requires
- **Complexity Estimate**: Initial complexity assessment (1-10 scale)
- **Category**: Core/Enhanced/Business/Innovation

## Output Requirements:

Generate comprehensive feature catalog including:
- 25-40 detailed feature specifications across all categories
- Features grounded in research findings
- Clear categorization and prioritization guidance
- Technical feasibility considerations
- User value assessments for each feature

**Success Criteria:**
- Comprehensive feature universe covering all major categories
- Features directly address research-validated user needs and market opportunities
- Clear specifications with implementation guidance
- Balanced mix of MVP, enhancement, and innovation features

Save results as: `feature_analysis/feature_universe_[TIMESTAMP].md`
```

### PHASE 4: Deploy Value & Scope Analysis Teams Simultaneously
You are about to deploy specialized evaluation agents to assess all generated features. Deploy both agents simultaneously using Task tool.

**Deploy Value Proposition Agent:**
Using the Task tool, spawn Value Proposition Agent with these complete instructions:

```
Task: Conduct strategic value assessment for [PROJECT_CONCEPT] features

💎 **VALUE_ANALYSIS EXECUTING** - Analyzing strategic value for [PROJECT_CONCEPT] features

## Task: Strategic Value Assessment

You are a Value Proposition Analyst assessing the strategic value of generated features.

**Project Context:**
- Project: [PROJECT_CONCEPT]
- Features: [Generated feature universe from previous phase]
- Research: [Market and user research findings]

## Value Assessment Methodology:

### 1. Strategic Alignment Analysis
- How well does each feature advance core project objectives?
- Does the feature directly address validated user needs?
- Does the feature create competitive advantage in the market?
- How essential is the feature to project success?

### 2. Business Impact Assessment
- Revenue generation potential
- User acquisition and retention impact
- Market differentiation value
- Cost-benefit analysis for development investment

### 3. User Value Delivery Analysis
- Direct user benefit and value creation
- User experience improvement impact
- Problem-solving effectiveness
- User adoption likelihood

### 4. Synergy & Integration Analysis
- How well does the feature work with other features?
- Does it amplify value of other capabilities?
- Integration complexity and value multiplication
- Ecosystem and platform benefits

## Value Scoring System:

Rate each feature 1-10 on:
- **Strategic Alignment** (1=tangential, 10=critical to mission)
- **Business Impact** (1=minimal impact, 10=major revenue/growth driver)
- **User Value** (1=nice to have, 10=essential user need)
- **Synergy Factor** (1=standalone, 10=amplifies other features)

**Overall Value Score**: Weighted average with strategic rationale

## Output Requirements:

Generate value assessment including:
- Individual feature value scores with detailed justification
- Value ranking of all features
- Strategic recommendations for feature prioritization
- Business case summary for highest-value features

**Success Criteria:**
- Comprehensive value assessment for all generated features
- Clear strategic rationale for scoring decisions
- Actionable prioritization guidance based on value analysis
- Business impact projections for key features

Save results as: `validation_reports/value_analysis_[TIMESTAMP].md`
```

**Deploy Technical Scope Agent:**
Using the Task tool, spawn Technical Scope Agent with these complete instructions:

```
Task: Conduct technical complexity assessment for [PROJECT_CONCEPT] features

🔧 **SCOPE_ANALYSIS EXECUTING** - Analyzing implementation complexity for [PROJECT_CONCEPT] features

## Task: Technical Complexity Assessment

You are a Technical Scope Analyst assessing implementation complexity and feasibility.

**Project Context:**
- Project: [PROJECT_CONCEPT]
- Domain: [DOMAIN_CONTEXT]
- Features: [Generated feature universe from previous phase]

## Complexity Assessment Methodology:

### 1. Implementation Complexity Analysis
- Development time and effort estimation
- Technical difficulty and skill requirements
- Architecture and infrastructure needs
- Testing and quality assurance complexity

### 2. Resource Requirements Assessment
- Team size and expertise requirements
- Development timeline estimates
- Infrastructure and technology stack needs
- Third-party dependencies and integrations

### 3. Risk & Feasibility Analysis
- Technical risks and implementation challenges
- Dependency risks and external factors
- Scalability and performance considerations
- Maintenance and support complexity

### 4. Integration Complexity Assessment
- How well does feature integrate with others?
- API and interface complexity
- Data flow and processing requirements
- System architecture impact

## Complexity Scoring System:

Rate each feature 1-10 on:
- **Development Complexity** (1=simple, 10=highly complex/risky)
- **Resource Requirements** (1=minimal resources, 10=major resource investment)
- **Integration Difficulty** (1=standalone, 10=complex system integration)
- **Technical Risk** (1=low risk, 10=high technical uncertainty)

**Overall Complexity Score**: Weighted average with technical rationale

## Output Requirements:

Generate complexity assessment including:
- Individual feature complexity scores with technical justification
- Complexity ranking of all features  
- Risk assessment and mitigation recommendations
- Resource and timeline estimates for implementation

**Success Criteria:**
- Comprehensive complexity assessment for all generated features
- Clear technical rationale for complexity scoring
- Actionable implementation guidance
- Risk mitigation strategies for complex features

Save results as: `validation_reports/scope_analysis_[TIMESTAMP].md`
```

**CRITICAL: Deploy both Value and Scope agents simultaneously. Do not conduct any analysis yourself.**

### PHASE 5: Decision Matrix Processing & Feature Pool Validation
- Apply decision matrix logic using Value and Complexity scores from agents
- **DO NOT score features yourself - use agent results**
- Identify features for MVP testing pool
- Deploy feature refinement agents if needed for problematic features

### PHASE 6: Deploy MVP Validation Teams Simultaneously
You are about to deploy MVP testing specialists to validate feature combinations. Deploy both agents simultaneously using Task tool.

**Deploy Market MVP Validation Agent:**
Using the Task tool, spawn Market MVP Validation Agent with these complete instructions:

```
Task: Conduct market validation testing for [PROJECT_CONCEPT] MVP combinations

🎯 **MVP_TESTING EXECUTING** - Market validation testing for [PROJECT_CONCEPT]

## Task: MVP Combination Market Validation

You are a Market Validation Specialist testing MVP combinations against market research.

**Context:**
- Validated feature pool: [Features that passed value/complexity decision matrix]
- Market research: [Findings from market analysis phase]

## MVP Testing Methodology:

### 1. MVP Combination Generation
- Create 8-10 different MVP feature combinations (3-5 features each)
- Focus on core value delivery with minimal complexity
- Ensure each MVP addresses fundamental user problems
- Balance market differentiation with implementation feasibility

### 2. Market Fit Assessment
For each MVP combination, evaluate:
- **Competitive Positioning**: How does this MVP compete in the market?
- **Market Differentiation**: What makes this MVP unique/better?
- **Market Timing**: Is the market ready for this solution?
- **Revenue Potential**: How well can this MVP generate revenue?
- **Market Penetration**: How easily can this MVP gain market share?

### 3. MVP Scoring System
Rate each MVP combination 1-10 on:
- **Market Differentiation** (1=commodity, 10=unique advantage)
- **Competitive Strength** (1=weak position, 10=market leader potential)
- **Revenue Potential** (1=limited monetization, 10=strong business model)
- **Market Readiness** (1=too early/late, 10=perfect timing)

## Output Requirements:

Generate market validation including:
- 8-10 tested MVP combinations with feature lists
- Market fit scores and detailed rationale for each MVP
- Ranking of MVPs by market opportunity
- Strategic recommendations for market positioning

**Success Criteria:**
- Comprehensive market testing of multiple MVP options
- Clear market opportunity assessment for each combination
- Data-driven recommendations for optimal market approach

Save results as: `validation_reports/mvp_market_validation_[TIMESTAMP].md`
```

**Deploy User MVP Validation Agent:**
Using the Task tool, spawn User MVP Validation Agent with these complete instructions:

```
Task: Conduct user validation testing for [PROJECT_CONCEPT] MVP combinations

👤 **MVP_TESTING EXECUTING** - User validation testing for [PROJECT_CONCEPT]

## Task: MVP Combination User Validation

You are a User Validation Specialist testing MVP combinations against user research.

**Context:**
- Validated feature pool: [Features that passed value/complexity decision matrix]
- User research: [Findings from user analysis phase]

## MVP Testing Methodology:

### 1. MVP Combination Assessment
- Test the same 8-10 MVP combinations as Market Validation agent
- Focus on user needs satisfaction and adoption likelihood
- Evaluate user experience and workflow effectiveness
- Assess learning curve and usability for target users

### 2. User Satisfaction Assessment
For each MVP combination, evaluate:
- **User Need Coverage**: How well does this MVP address core user problems?
- **User Experience Quality**: How intuitive and effective is the user workflow?
- **Adoption Likelihood**: How likely are users to adopt this MVP?
- **User Value Delivery**: How much value does this create for users?
- **Learning Curve**: How easy is it for users to get value from this MVP?

### 3. MVP Scoring System
Rate each MVP combination 1-10 on:
- **User Need Satisfaction** (1=doesn't solve problem, 10=perfectly addresses needs)
- **Usability & Experience** (1=poor UX, 10=excellent user experience)
- **Adoption Probability** (1=users won't adopt, 10=high adoption likelihood)
- **Value Creation** (1=minimal user value, 10=high user value creation)

## Output Requirements:

Generate user validation including:
- User satisfaction scores and detailed rationale for each MVP
- Ranking of MVPs by user value and adoption potential
- User experience recommendations for top MVP combinations
- User validation testing recommendations

**Success Criteria:**
- Comprehensive user testing of multiple MVP options
- Clear user satisfaction assessment for each combination
- Data-driven recommendations for optimal user experience

Save results as: `validation_reports/mvp_user_validation_[TIMESTAMP].md`
```

**CRITICAL: Deploy both MVP validation agents simultaneously. Do not conduct any validation yourself.**

### PHASE 7: MVP Selection & ProductBrief Synthesis
- Coordinate agent results compilation (DO NOT analyze results yourself)
- Apply MVP selection algorithm based on agent findings
- Synthesize all agent research into final ProductBrief.json
- **Your role is coordination and compilation, not analysis**

## Orchestrator Responsibilities (What You DO)

✅ **Deploy specialized agents using Task tool**
✅ **Monitor agent completion and progress** 
✅ **Coordinate handoffs between phases**
✅ **Compile agent results into ProductBrief.json**
✅ **Apply systematic decision frameworks using agent data**
✅ **Report workflow status and progress**

## What You DO NOT Do

❌ **Conduct market research yourself**
❌ **Analyze user needs yourself** 
❌ **Generate features yourself**
❌ **Score or evaluate features yourself**
❌ **Validate MVP combinations yourself**
❌ **Any domain expertise work - deploy agents for that**

## Success Criteria

- All required specialist agents deployed successfully using Task tool
- Parallel execution achieved for research phases
- Complete ProductBrief.json generated from agent findings
- Validated MVP selected through systematic agent testing
- No research or analysis conducted by orchestrator directly

**You are the conductor of a research orchestra - coordinate the musicians, don't play the instruments yourself.**"
```