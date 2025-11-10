
### 2.05 Create WBS (Work Breakdown Structure)

**Inputs:**
- Project Scope Statement
- Requirements Documentation

**Tools & Techniques:**

**Decomposition:**

**WBS STRUCTURE**

```
Level 1: PROJECT
Customer Pain Point Discovery Agent

Level 2: PHASES
├── 1.0 Project Management
├── 2.0 Phase 1: Foundation 
├── 3.0 Phase 2: Multi-Source Integration 
├── 4.0 Phase 3: Enhancement & Polish
└── 5.0 Phase 4: Deployment & Documentation

Level 3: DELIVERABLES

1.0 PROJECT MANAGEMENT
├── 1.1 Project Planning
├── 1.2 Team Coordination
├── 1.3 Status Reporting
└── 1.4 Risk Management

2.0 PHASE 1: FOUNDATION 
├── 2.1 Development Environment Setup
├── 2.2 Reddit Integration
├── 2.3 LangChain Agent Core
└── 2.4 Basic Streamlit UI

3.0 PHASE 2: MULTI-SOURCE INTEGRATION 
├── 3.1 Twitter/X API Integration
├── 3.2 Google Search API Integration
├── 3.3 Multi-Tool Agent Orchestration
└── 3.4 Enhanced UI (filters, export)

4.0 PHASE 3: ENHANCEMENT & POLISH
├── 4.1 Error Handling & Retry Logic
├── 4.2 Performance Optimization
├── 4.3 UI/UX Improvements
└── 4.4 Comprehensive Testing

5.0 PHASE 4: DEPLOYMENT & DOCUMENTATION
├── 5.1 Production Deployment
├── 5.2 Documentation
├── 5.3 Demo Video Creation
└── 5.4 Team Knowledge Transfer

Level 4: WORK PACKAGES

2.1 Development Environment Setup
├── 2.1.1 GitHub Repository Creation
├── 2.1.2 Python Environment Configuration
├── 2.1.3 API Key Provisioning
└── 2.1.4 Dependency Installation

2.2 Reddit Integration
├── 2.2.1 Reddit API Wrapper Development
├── 2.2.2 Reddit Search Functionality
├── 2.2.3 Data Parsing & Formatting
└── 2.2.4 Unit Tests for Reddit Tool

2.3 LangChain Agent Core
├── 2.3.1 Agent Initialization
├── 2.3.2 Tool Registration
├── 2.3.3 Query Processing Logic
└── 2.3.4 Pain Point Extraction with LLM

2.4 Basic Streamlit UI
├── 2.4.1 UI Layout Design
├── 2.4.2 Query Input Component
├── 2.4.3 Results Display Component
└── 2.4.4 Basic Styling

(Continue for all phases...)
```

**Outputs:**

**SCOPE BASELINE**

The Scope Baseline consists of:
1. Project Scope Statement (above)
2. WBS (above)
3. WBS Dictionary (below)

**WBS DICTIONARY** (Sample Entries)

```
WORK PACKAGE: 2.2 Reddit Integration

Code: 2.2
Description: Develop tool to search Reddit for customer pain point discussions
Deliverable: Functional Reddit API wrapper that returns structured post data

Acceptance Criteria:
- Searches 3+ subreddits simultaneously
- Returns 10-20 relevant posts per query
- Handles API errors gracefully
- Returns data in standardized format (title, text, upvotes, url)

Owner: ?
Duration: 2 days (16 hours)
Dependencies: 2.1 (Environment Setup) must be complete
Resources: 1 developer, Reddit API documentation
Cost: $0 (Reddit API is free)

---

WORK PACKAGE: 2.3 LangChain Agent Core

Code: 2.3
Description: Build agent orchestration logic using LangChain framework
Deliverable: Working agent that can select tools and process queries

Acceptance Criteria:
- Agent accepts natural language queries
- Agent selects appropriate tool(s) based on query
- Agent chains tool results
- Agent passes results to LLM for extraction

Owner: ?
Duration: 2 days (16 hours)
Dependencies: 2.2 (Reddit Integration) must be complete
Resources: 1 developer, LangChain documentation, OpenAI API
Cost: $2 (OpenAI API testing)

---

WORK PACKAGE: 3.1 Twitter/X API Integration

Code: 3.1
Description: Integrate Twitter API for social media pain point discovery
Deliverable: Twitter API wrapper returning tweet data

Acceptance Criteria:
- Searches Twitter for relevant tweets
- Returns 10-20 tweets per query
- Parses tweet text, author, engagement
- Handles rate limiting

Owner: ?
Duration: 1.5 days (12 hours)
Dependencies: 2.2 (Reddit Integration) as template
Resources: 1 developer, Twitter API docs
Cost: $0-5 (Twitter API basic tier)
```

---

## 3. SCHEDULE MANAGEMENT

### 2.06 Plan Schedule Management

**Outputs:**

**SCHEDULE MANAGEMENT PLAN**

```
HOW SCHEDULE WILL BE PLANNED:
- Work packages estimated in hours (not story points)
- Each developer allocated 2-3 hours/day (10-15 hours/week)
- Product Owner and TPM: 5-10 hours/week
- Buffer time: 20% added to each work package estimate

HOW SCHEDULE WILL BE EXECUTED:
- Daily asynchronous updates in Slack/Discord
- Weekly 1-hour synchronous team meeting
- Pair programming for complex tasks
- GitHub Projects board for task tracking

HOW SCHEDULE WILL BE CONTROLLED:
- Daily progress check by TPM
- Weekly sprint retrospective
- Variance analysis (actual vs. planned hours)
- Corrective action if >20% variance on critical path

LEVELS OF ACCURACY:
- Phase-level estimates: ±25% accuracy
- Work package estimates: ±15% accuracy
- Activity estimates: ±10% accuracy

RULES OF PERFORMANCE MEASUREMENT:
- Earned Value Management (simplified)
- Tasks marked as 0%, 50%, or 100% complete
- No partial credit for incomplete work packages

PROJECT SCHEDULE MODEL DEVELOPMENT:
- Precedence Diagramming Method (PDM)
- Critical Path Method (CPM)
- Rolling wave planning (Week 1 detailed, Weeks 2-4 high-level)

REPORTING FORMATS:
- Weekly status report (narrative)
- Gantt chart (visual timeline)
- Burn-down chart (tasks remaining)

RELEASE AND ITERATION LENGTH:
- Sprint length: 1 week
- Release cycle: Phase completion (every week)
- Final release: Week 4 (Dec 6, 2025)
```

---

### 2.07 Define Activities

**Inputs:**
- WBS (from 2.05)

**Tools & Techniques:**

**Decomposition:**
Work packages from WBS broken down into scheduled activities.

**Rolling Wave Planning:**
- Week 1 (Nov 11-15): Detailed activity planning completed
- Week 2 (Nov 18-22): High-level activities defined, details added during Week 1
- Week 3 (Nov 25-29): High-level activities defined, details added during Week 2
- Week 4 (Dec 2-6): High-level activities defined, details added during Week 3

**Outputs:**

**ACTIVITY LIST (Week 1 - Detailed)**

```
PHASE 1: FOUNDATION (Week 1: Nov 11-15)

Activity ID: 1.1.1
Activity: Create GitHub Repository
Description: Initialize repo with folder structure, README template, .gitignore
Work Package: 2.1.1 GitHub Repository Creation
Duration: 0.5 hours
Owner: Nicolai

Activity ID: 1.1.2
Activity: Set up Python virtual environment
Description: Create venv, install dependencies from requirements.txt
Work Package: 2.1.2 Python Environment Configuration
Duration: 0.5 hours
Owner: All team members (parallel)

Activity ID: 1.1.3
Activity: Obtain OpenAI API key
Description: Sign up, generate API key, test with simple prompt
Work Package: 2.1.3 API Key Provisioning
Duration: 0.25 hours
Owner: All team members

Activity ID: 1.1.4
Activity: Obtain Reddit API credentials
Description: Create Reddit app, get client_id and client_secret
Work Package: 2.1.3 API Key Provisioning
Duration: 0.25 hours
Owner: All team members

Activity ID: 1.1.5
Activity: Test API connections
Description: Run test scripts to verify OpenAI and Reddit APIs working
Work Package: 2.1.4 Dependency Installation
Duration: 0.5 hours
Owner: Javier, Stefan

Activity ID: 1.2.1
Activity: Design Reddit API wrapper interface
Description: Define function signatures and return data structure
Work Package: 2.2.1 Reddit API Wrapper Development
Duration: 1 hour
Owner: Edison

Activity ID: 1.2.2
Activity: Implement Reddit search functionality
Description: Code search_subreddits() function with error handling
Work Package: 2.2.2 Reddit Search Functionality
Duration: 4 hours
Owner: Edison

Activity ID: 1.2.3
Activity: Implement Reddit data parsing
Description: Parse Reddit API response into standardized format
Work Package: 2.2.3 Data Parsing & Formatting
Duration: 2 hours
Owner: Edison

Activity ID: 1.2.4
Activity: Write unit tests for Reddit tool
Description: Test cases for search, parsing, error conditions
Work Package: 2.2.4 Unit Tests for Reddit Tool
Duration: 2 hours
Owner: Amanda

Activity ID: 1.3.1
Activity: Design pain point extraction prompt
Description: Write LLM prompt for extracting pain points from text
Work Package: 2.3.4 Pain Point Extraction with LLM
Duration: 1 hour
Owner: Javier

Activity ID: 1.3.2
Activity: Implement pain point extractor
Description: Create extract_pain_points() function using OpenAI API
Work Package: 2.3.4 Pain Point Extraction with LLM
Duration: 3 hours
Owner: Javier

Activity ID: 1.3.3
Activity: Initialize LangChain agent
Description: Set up agent with ChatOpenAI and tool list
Work Package: 2.3.1 Agent Initialization
Duration: 2 hours
Owner: Stefan

Activity ID: 1.3.4
Activity: Register tools with agent
Description: Create Tool wrappers for Reddit search and extraction
Work Package: 2.3.2 Tool Registration
Duration: 2 hours
Owner: Stefan

Activity ID: 1.3.5
Activity: Implement agent query processing
Description: Create run() method that orchestrates tool calls
Work Package: 2.3.3 Query Processing Logic
Duration: 3 hours
Owner: Stefan

Activity ID: 1.4.1
Activity: Design Streamlit UI layout
Description: Sketch UI wireframes, define components
Work Package: 2.4.1 UI Layout Design
Duration: 1 hour
Owner: Al

Activity ID: 1.4.2
Activity: Implement query input component
Description: Create text input and button in Streamlit
Work Package: 2.4.2 Query Input Component
Duration: 1 hour
Owner: Al

Activity ID: 1.4.3
Activity: Implement results display component
Description: Format and display agent output in Streamlit
Work Package: 2.4.3 Results Display Component
Duration: 2 hours
Owner: Al

Activity ID: 1.4.4
Activity: Add basic styling
Description: Apply colors, fonts, layout improvements
Work Package: 2.4.4 Basic Styling
Duration: 1 hour
Owner: Al

Activity ID: 1.5.1
Activity: End-to-end testing
Description: Test complete flow from query to results
Work Package: 2.4 (Deliverable acceptance)
Duration: 1 hour
Owner: Amanda

Activity ID: 1.5.2
Activity: Code review
Description: Review all code for quality and documentation
Work Package: 2.4 (Deliverable acceptance)
Duration: 2 hours
Owner: Nicolai

Activity ID: 1.5.3
Activity: Push to GitHub
Description: Commit all code with descriptive messages
Work Package: 2.4 (Deliverable acceptance)
Duration: 0.5 hours
Owner: Nicolai
```

**ACTIVITY ATTRIBUTES**

```
Activity ID: 1.2.2
Activity Name: Implement Reddit search functionality
Point of Contact: ?
Location of Work: Remote (individual work)
Calendar: Working days (Mon-Fri)
Work Hours: 9 AM - 5 PM EST (flexible)
Resource Requirements: 1 developer, Reddit API docs, IDE
Predecessor Activities: 1.2.1 (Design Reddit API wrapper)
Successor Activities: 1.2.3 (Implement Reddit data parsing)
Assumptions: Reddit API is accessible and documented
Constraints: Must handle rate limiting (60 requests/minute)
Level of Effort: 4 hours
Coding Requirements: Python 3.9+, praw library
Testing Requirements: Unit tests with mock data
```

**MILESTONE LIST**

```
Milestone ID: M1
Milestone Name: Phase 1 Complete - Basic Agent Working
Description: Agent can search Reddit and extract pain points via Streamlit UI
Acceptance Criteria:
- Reddit tool returns 10+ posts
- LLM extraction returns structured pain points
- Streamlit displays results
- Code pushed to GitHub

---

Milestone ID: M2
Milestone Name: Phase 2 Complete - Multi-Source Agent
Description: Agent searches 3 sources (Reddit, Twitter, Google)
Acceptance Criteria:
- All 3 tools integrated
- Agent orchestrates multiple sources
- Results aggregated and de-duplicated
- Enhanced UI with filters

---

Milestone ID: M3
Milestone Name: Phase 3 Complete - Production-Ready
Description: Error handling, performance optimization, comprehensive testing
Acceptance Criteria:
- Error handling for all API failures
- Response time < 2 minutes
- 10+ test cases passing
- UI polished and mobile-responsive

---

Milestone ID: M4
Milestone Name: Project Complete - Deployed & Documented
Description: Live deployment, documentation, demo video
Acceptance Criteria:
- Deployed on Streamlit Cloud (public URL)
- README with setup instructions
- Demo video created and uploaded
- All team members can demo the project

---

Milestone ID: M5
Milestone Name: 50% Code Complete
Type: Optional (Progress Indicator)
Description: Half of all planned code written
Criteria: 50% of work packages marked complete

---

Milestone ID: M6
Milestone Name: First Public Demo
Type: Optional (Marketing)
Description: Share demo link on LinkedIn/Twitter
Criteria: Agent functional enough for external viewing
```

**Recommendation for Monday:**
1. Review and approve Project Charter + Scope Statement (15 min)
2. Validate requirements with team (10 min)
