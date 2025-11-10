# Customer Pain Point Discovery Agent - Project Management

---

## 1. INTEGRATION MANAGEMENT

**PROJECT CHARTER**

```
PROJECT TITLE: Customer Pain Point Discovery Agent

PROJECT PURPOSE/JUSTIFICATION:
Build an AI agent that automatically discovers customer pain points from multiple online sources 
(Reddit, Twitter, web search) to demonstrate modern AI/ML skills for job applications.

PROJECT DESCRIPTION:
An intelligent agent using LangChain and OpenAI that:
- Accepts user query (e.g., "What pain points do B2B product managers face?")
- Searches multiple data sources (Reddit, Twitter, Google)
- Extracts and categorizes pain points using LLM
- Presents structured findings in Streamlit dashboard

HIGH-LEVEL REQUIREMENTS:
1. Agent must search at least 3 data sources
2. Output must be structured (JSON/table format)
3. Must include source citations
4. Response time < 2 minutes
5. Streamlit UI must be intuitive for non-technical users
6. Deployed publicly (Streamlit Cloud)
7. Full documentation on GitHub

SUCCESS CRITERIA:
- Agent correctly identifies pain points from 10 test queries (80%+ accuracy)
- All team members can demo the project
- Code reviewed and documented
- Live URL shareable in resumes/LinkedIn

HIGH-LEVEL RISKS:
- API rate limits (Reddit, Twitter, OpenAI)
- Team members getting jobs mid-project
- API costs exceeding budget ($50 limit)
- Complexity of multi-tool agent orchestration

SUMMARY MILESTONE SCHEDULE:
- Phase 1 - Reddit integration + basic agent
- Phase 2 - Add Twitter + Google Search
- Phase 3 - UI polish + error handling
- Phase 4 - Deployment + documentation

SUMMARY BUDGET:
- OpenAI API: $10-15
- Reddit API: Free
- Twitter API: $0-5 (basic tier)
- Google Search API: $0 (100 queries/day free)
- Streamlit Cloud: Free
- Total: $10-20

PROJECT APPROVAL REQUIREMENTS:
Team consensus on major decisions (scope, tools, timeline)

```

**ASSUMPTION LOG**

```
ID | Assumption | Impact | Owner | Status
1 | All team members have 10-20 hrs/week available | High | Nicolai | Valid
2 | OpenAI API costs stay under $20 | Medium | Nicolai | Valid
3 | APIs (Reddit, Twitter, OpenAI) remain accessible | High | Nicolai | Valid
4 | No team members get jobs before Week 2 | Medium | All | Monitor
5 | Python 3.9+ environment works for all team members | Low | Nicolai | Valid
6 | Streamlit Cloud free tier sufficient for deployment | Low | Al | Valid
```

---

### 2.03 Collect Requirements

**Context Diagram:**
```
┌─────────────┐
│    User     │
│ (PM/Analyst)│
└──────┬──────┘
       │ Query: "What are pain points?"
       ↓
┌─────────────────────────────────────────┐
│  Customer Pain Point Discovery Agent    │
│                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │ Reddit  │  │ Twitter │  │ Google  ││
│  │  Tool   │  │  Tool   │  │  Tool   ││
│  └─────────┘  └─────────┘  └─────────┘│
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   LLM Pain Point Extractor        │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Streamlit Dashboard             │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
       │
       ↓ Structured Report
┌──────────────┐
│  Output:     │
│  - Pain pts  │
│  - Sources   │
│  - Frequency │
└──────────────┘
```

**Data Gathering: Benchmarking**
- Compared to similar projects: LangChain agent examples, AutoGPT
- Reference: https://github.com/langchain-ai/langchain-examples


**REQUIREMENTS DOCUMENTATION**

```
FUNCTIONAL REQUIREMENTS:

FR-1: Agent Query Processing
Description: System must accept natural language queries about customer pain points
Acceptance Criteria:
- User can input query via text box
- System processes queries with 1-50 words
- System returns error for empty queries
Priority: Must Have
Stakeholder: Al (Product Owner), Users

FR-2: Multi-Source Data Collection
Description: Agent must search at least 3 data sources
Acceptance Criteria:
- Reddit API integration working
- Twitter API integration working
- Google Search API integration working
- Each source returns 5-10 relevant results
Priority: Must Have
Stakeholder: Nicolai (Technical PM)

FR-3: Pain Point Extraction
Description: System extracts and categorizes pain points using LLM
Acceptance Criteria:
- LLM identifies 3-5 pain points per query
- Each pain point has: name, description, frequency, example quote
- Output formatted as structured JSON
- Accuracy: 80%+ on test queries
Priority: Must Have
Stakeholder: Al (Product Owner)

FR-4: Source Citation
Description: Every pain point must cite source(s)
Acceptance Criteria:
- Each pain point shows source URL
- Source includes: platform (Reddit/Twitter), timestamp, author
- User can click to view original post
Priority: Should Have
Stakeholder: Amanda (QA)

FR-5: Streamlit Dashboard
Description: Web interface for inputting queries and viewing results
Acceptance Criteria:
- Clean, intuitive UI (non-technical users can use)
- Results displayed within 5 seconds of API completion
- Mobile-responsive design
Priority: Must Have
Stakeholder: Al (Product Owner)

NON-FUNCTIONAL REQUIREMENTS:

NFR-1: Performance
- Response time: < 2 minutes for standard query
- Support 100+ concurrent queries (Streamlit Cloud limit)
Priority: Should Have

NFR-2: Reliability
- 95% uptime (dependent on Streamlit Cloud)
- Graceful handling of API failures
- No data loss if API times out
Priority: Should Have

NFR-3: Scalability
- Support 3 data sources initially
- Architecture allows adding 5+ more sources
Priority: Should Have

NFR-4: Maintainability
- Code documented with docstrings
- README includes setup instructions
- Modular architecture (easy to swap components)
Priority: Must Have

NFR-5: Cost
- Total API costs < $20 for 4-week project
- OpenAI costs < $15
Priority: Must Have

CONSTRAINTS:
- Budget: $20 maximum
- Timeline: 4 weeks
- Team availability: 10-20 hours/week per person
- Technical: Must use Python, LangChain, OpenAI
```



### 2.04 Define Scope

**Product Analysis:**
```
SYSTEM ANALYSIS:
Input: Natural language query
Processing: Multi-tool agent orchestration → LLM extraction → Formatting
Output: Structured pain point report with sources

PRODUCT BREAKDOWN:
1. Agent Core (LangChain framework)
   - Query parser
   - Tool selector
   - Response aggregator
   
2. Data Collection Tools
   - Reddit API wrapper
   - Twitter API wrapper
   - Google Search API wrapper
   
3. LLM Processing
   - Pain point extraction
   - Categorization
   - Sentiment analysis (optional)
   
4. User Interface
   - Streamlit dashboard
   - Result visualization
   - Export functionality
   
5. Infrastructure
   - GitHub repository
   - Streamlit Cloud deployment
   - API key management
```


**PROJECT SCOPE STATEMENT**

```
PRODUCT SCOPE DESCRIPTION:
An AI-powered agent that automatically discovers and categorizes customer pain points 
by searching multiple online sources (Reddit, Twitter, Google) and using LLM analysis 
to extract structured insights.

PROJECT SCOPE DESCRIPTION:
The work required to deliver the pain point discovery agent includes:
- Setting up development environment and APIs
- Building modular data collection tools for 3+ sources
- Integrating LangChain agent framework
- Implementing LLM-based extraction logic
- Creating Streamlit web interface
- Writing comprehensive documentation
- Deploying to production (Streamlit Cloud)
- Conducting end-to-end testing

PROJECT DELIVERABLES:
1. Working AI agent (Python codebase)
2. Streamlit web application
3. GitHub repository with documentation
4. Live demo URL (Streamlit Cloud)
5. Test suite (10+ test cases)
6. User guide (README.md)
7. Technical architecture document
8. Demo video (2-3 minutes)

ACCEPTANCE CRITERIA:
- Agent successfully searches 3 data sources
- Pain point extraction accuracy ≥ 80% (10 test queries)
- Response time < 2 minutes
- Streamlit UI functional on desktop and mobile
- Code documented with docstrings
- All APIs working (Reddit, Twitter, OpenAI, Google)
- Deployed and accessible via public URL
- Zero critical bugs remaining

PROJECT EXCLUSIONS (OUT OF SCOPE):
- Custom-trained ML models (using pre-trained only)
- Real-time streaming data (batch processing only)
- User authentication/accounts
- Data storage/database (ephemeral results only)
- Multi-language support (English only)
- Mobile native app (web only)
- Advanced analytics dashboard
- Automated email reports
- Integration with third-party tools (Slack, email)

CONSTRAINTS:
- Timeline: 1-2 weeks
- Budget: $20 API costs
- Team: 7 people at 10-20 hours/week each
- Technology: Must use Python, LangChain, OpenAI
- APIs: Dependent on third-party availability
- Deployment: Free tier only (Streamlit Cloud)

ASSUMPTIONS:
- All team members available 
- API rate limits sufficient for development
- Streamlit Cloud performance adequate
- No major API changes during project

PROJECT RISKS:
- High: Team members getting jobs mid-project
- Medium: API costs exceeding budget
- Medium: API rate limiting blocking development
- Low: Technical complexity beyond team skill level
```

3. Confirm Week 1 activity assignments (10 min)
4. Estimate durations for Week 1 activities (15 min)
5. Create Week 1 schedule with start/finish dates (10 min)

Total meeting time: 60 minutes
```
