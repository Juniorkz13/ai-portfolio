# Legal RAG Multi-Agent 🏛️⚖️

An intelligent multi-domain legal analysis system powered by **LangGraph**, **Google Gemini 2.0**, and **FastAPI**. This system leverages advanced AI agents to provide preliminary legal guidance across multiple practice areas of Brazilian law.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Author](#author)

## 🎯 Problem Statement

### The Challenge

Millions of people worldwide face legal questions but lack access to affordable legal counsel. In Brazil, this problem is particularly acute:

- **High Cost Barrier**: Legal consultations are expensive, placing them out of reach for many citizens
- **Information Gap**: Citizens struggle to understand their rights and legal obligations
- **Accessibility Issues**: Legal expertise is concentrated in major urban centers, leaving rural and remote areas underserved
- **Time Constraints**: Legal professionals have limited availability to handle preliminary consultations
- **Complexity**: Brazilian law (CLT, Civil Code, Consumer Code, etc.) is complex and constantly evolving

Citizens need:
- **Quick answers** to common legal questions
- **Reliable information** about their rights and obligations
- **Guidance** on what to do next
- **Risk assessment** to understand the severity of their situation
- **24/7 access** without geographic limitations

### Current Solutions' Limitations

1. **Generic Legal Websites**: Provide surface-level information without context analysis
2. **ChatGPT-Based Systems**: Lack specialized legal knowledge and risk awareness
3. **Traditional Legal Advice**: Expensive and time-consuming
4. **Government Portals**: Often outdated and difficult to navigate

## 💡 Solution Overview

**Legal RAG Multi-Agent** is an intelligent legal guidance system that addresses these challenges through:

### How It Works

```
User Question
    ↓
1. ANALYZE: Identify the legal domain (Labor, Civil, Consumer, etc.)
    ↓
2. PLAN: Generate optimized search queries
    ↓
3. RETRIEVE: Gather relevant legal documents and references
    ↓
4. VERIFY: Check for legal conflicts and inconsistencies
    ↓
5. INTERPRET: Detect ambiguities and missing information
    ↓
6. ASSESS: Evaluate risk level based on complexity
    ↓
7. GENERATE: Create comprehensive, personalized legal guidance
    ↓
Detailed Legal Analysis + Risk Assessment + Actionable Recommendations
```

### What Makes It Different

1. **Multi-Agent Architecture**: Six specialized AI agents work together, each handling a specific aspect of legal analysis
2. **GenAI-Powered**: Uses Google Gemini 2.0 Flash Latest for generating contextual, nuanced legal guidance (not templated responses)
3. **Risk-Aware**: Automatically detects conflicting legal positions and ambiguous situations
4. **Domain-Specific**: Understands context within specific legal domains (Labor Law, Consumer Law, etc.)
5. **Preliminary Analysis**: Clearly marks itself as a starting point, always recommending professional legal consultation
6. **Multi-Language Ready**: Built to support Portuguese (pt-BR) with easy extension to other languages

### The Value Proposition

- ✅ **24/7 Availability**: Get legal guidance anytime, anywhere
- ✅ **Cost-Free**: No expensive consultation fees for preliminary analysis
- ✅ **Fast Response**: Instant analysis instead of weeks for legal appointments
- ✅ **Comprehensive**: Covers multiple legal domains
- ✅ **Risk Transparency**: Clear communication of risk levels and limitations
- ✅ **Empowering**: Users understand their rights and can prepare before consulting a lawyer

## 🎯 Key Features

### 🤖 Multi-Agent Architecture

Six specialized legal AI agents work in harmony:

1. **Query Planner Agent**: Breaks down user questions into optimized search queries
2. **Retriever Agent**: Finds relevant legal documents and case references
3. **Cross-Reference Agent**: Identifies conflicts between different legal sources
4. **Legal Interpreter Agent**: Determines legal domain and detects ambiguities
5. **Risk Assessment Agent**: Evaluates complexity and legal risks
6. **Answer Agent**: Generates comprehensive legal guidance using Gemini AI

### 🧠 Intelligent Legal Guidance

- Powered by Google Gemini 2.0 Flash Latest
- Contextual responses based on user's specific situation
- References to Brazilian legislation (CLT, Civil Code, Consumer Code, etc.)
- Real-time risk assessment

### 🔐 Enterprise Security

- **JWT Authentication**: Secure token-based API access
- **API Key Validation**: Additional layer of API security
- **Rate Limiting**: Protection against abuse (100 requests/hour per user)
- **Structured Logging**: Complete audit trail of all analyses

### 📊 Intelligent Risk Assessment

- Automatic detection of legal conflicts
- Identification of ambiguous questions
- Assessment of information gaps
- Color-coded risk levels: 🟢 Low / 🟡 Medium / 🔴 High
- Confidence scoring for each analysis

### 📚 Multi-Domain Legal Coverage

Supports analysis across 9 major legal domains:
- 🏢 **Labor Law** (CLT): Employment contracts, termination, benefits
- 📋 **Civil Law**: Contracts, liability, damages
- 🛍️ **Consumer Law**: Product defects, warranty rights
- 👨‍👩‍👧 **Family Law**: Divorce, custody, child support
- ⚡ **Criminal Law**: Crime classification, procedure
- 💰 **Tax Law**: Income tax, ICMS, contributions
- 🏭 **Business Law**: Company formation, corporate matters
- 🏦 **Social Security Law**: Retirement, INSS, benefits
- 🏠 **Real Estate Law**: Leasing, property ownership

### 📖 API Documentation

- OpenAPI/Swagger UI at `/docs`
- Interactive API testing
- Auto-generated schema documentation

### 🧪 Comprehensive Testing

- 16 unit tests with 100% pass rate
- Coverage across all legal agents
- Authentication and authorization tests
- Rate limiting validation

## 🏗️ Architecture

### Complete System Design

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            FULL-STACK ARCHITECTURE                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FRONTEND LAYER (React 19 + TypeScript + Vite)                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │ │
│  │  │  Login Page      │  │ Dashboard Page   │  │  History Sidebar │         │ │
│  │  │                  │  │                  │  │                  │         │ │
│  │  │ • Form Validation│  │ • Text Analysis  │  │ • Recent Queries │         │ │
│  │  │ • JWT Storage    │  │ • Risk Display   │  │ • Clear History  │         │ │
│  │  │ • Auth Flow      │  │ • Results Panel  │  │ • Timestamps     │         │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘         │ │
│  │                                 ↓                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  │ State Management (Zustand)                                             │ │
│  │  │ • authStore (JWT token, user data)                                    │ │
│  │  │ • historyStore (previous analyses)                                    │ │
│  │  │ • Persistence (localStorage)                                          │ │
│  │  └────────────────────────────────────────────────────────────────────────┘ │
│  │                                 ↓                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  │ HTTP Client (Axios)                                                    │ │
│  │  │ • Base configuration                                                   │ │
│  │  │ • Request interceptors (Token injection)                               │ │
│  │  │ • Response interceptors (401 handling)                                 │ │
│  │  │ • Error handling                                                       │ │
│  │  └────────────────────────────────────────────────────────────────────────┘ │
│  │                                 ↓                                            │ │
│  │  PUBLISHED: Vercel | DEVELOPMENT: Vite Dev Server (Port 3000)               │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                   ↓↓↓                                            │
│                          (HTTP/HTTPS Communication)                             │
│                                   ↓↓↓                                            │
│  BACKEND LAYER (FastAPI + LLM Agents)                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │              API Layer (FastAPI)                                     │ │ │
│  │  │  • POST /api/v1/login (Authentication)                              │ │ │
│  │  │  • POST /api/v1/analyze (Legal Analysis)                            │ │ │
│  │  │  • GET /api/v1/models (Available Models)                            │ │ │
│  │  │  • GET /health (Health Check)                                       │ │ │
│  │  │  • JWT verification, Rate limiting, CORS middleware                 │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  │                           ↓                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │         Multi-Agent Legal Workflow (Sequential Orchestration)       │ │ │
│  │  │                                                                      │ │ │
│  │  │  1. Query Planner     → Decompose question into search queries      │ │ │
│  │  │          ↓                                                           │ │ │
│  │  │  2. Legal Interpreter → Identify domain and ambiguities             │ │ │
│  │  │          ↓                                                           │ │ │
│  │  │  3. Retriever Agent   → Find relevant documents                     │ │ │
│  │  │          ↓                                                           │ │ │
│  │  │  4. Cross Reference   → Detect legal conflicts                      │ │ │
│  │  │          ↓                                                           │ │ │
│  │  │  5. Risk Assessor     → Evaluate complexity and risks               │ │ │
│  │  │          ↓                                                           │ │ │
│  │  │  6. Answer Agent      → Generate final response with disclaimers    │ │ │
│  │  │                                                                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  │                           ↓                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │         Gen AI Layer (Google Gemini 2.0 Flash Latest)              │ │ │
│  │  │  • Natural Language Understanding                                   │ │ │
│  │  │  • Legal Context Analysis                                          │ │ │
│  │  │  • Answer Generation & Formatting                                  │ │ │
│  │  │  • Risk Assessment Integration                                     │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  │                           ↓                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │         Response Formatting                                         │ │ │
│  │  │  • Structured JSON output                                           │ │ │
│  │  │  • Risk levels + metrics                                            │ │ │
│  │  │  • Legal recommendations                                            │ │ │
│  │  │  • Comprehensive disclaimers                                        │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                            │ │
│  │  INFRASTRUCTURE:                                                           │ │
│  │  • Uvicorn ASGI Server (Port 8000)                                        │ │
│  │  • Logging & Request Tracking                                             │ │
│  │  • Error Handling & Recovery                                              │ │
│  │  • Docker Container Support                                               │ │
│  │                                                                            │ │
│  │  DEPLOYMENT: Docker | Render.com | Google Cloud Run | AWS Lambda         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
legal-rag-multi-agent/
│
├── 📱 FRONTEND (React 19 + TypeScript + Vite)
│   └── frontend/
│       ├── src/
│       │   ├── pages/
│       │   │   ├── LoginPage.tsx          # 🔐 Authentication UI
│       │   │   └── DashboardPage.tsx      # 📊 Main analysis interface
│       │   │
│       │   ├── components/
│       │   │   ├── Common/
│       │   │   │   └── Button.tsx         # Reusable button component
│       │   │   ├── Analysis/              # Analysis result components
│       │   │   ├── Auth/                  # Authentication components
│       │   │   ├── Chat/                  # Chat interface components
│       │   │   └── Layout/                # Layout wrapper components
│       │   │
│       │   ├── services/
│       │   │   └── api.ts                 # 🔗 Axios client + interceptors
│       │   │
│       │   ├── store/                     # State Management (Zustand)
│       │   │   ├── authStore.ts           # 👤 User auth state
│       │   │   └── historyStore.ts        # 📜 Query history
│       │   │
│       │   ├── hooks/                     # Custom React hooks
│       │   ├── types/
│       │   │   └── index.ts               # TypeScript interfaces
│       │   ├── utils/
│       │   │   └── cn.ts                  # Class name utilities
│       │   ├── assets/                    # Static assets
│       │   ├── App.tsx                    # Root component
│       │   ├── main.tsx                   # Entry point
│       │   └── index.css                  # Global styles
│       │
│       ├── public/                        # Static files
│       ├── package.json                   # Dependencies
│       ├── vite.config.ts                 # Vite configuration
│       ├── tailwind.config.js             # Tailwind CSS config
│       ├── tsconfig.json                  # TypeScript config
│       └── vercel.json                    # Vercel deployment config
│
├── 🐍 BACKEND (FastAPI + Python)
│   ├── app/
│   │   ├── agents/                       # 🤖 AI Agents
│   │   │   ├── base.py                   # Abstract base class
│   │   │   ├── query_planner.py          # Query decomposition
│   │   │   ├── retriever.py              # Document retrieval
│   │   │   ├── cross_reference.py        # Conflict detection
│   │   │   ├── legal_interpreter.py      # Domain identification
│   │   │   ├── risk.py                   # Risk assessment
│   │   │   ├── answer.py                 # Answer generation
│   │   │   └── answer_agent.py           # Answer orchestration
│   │   │
│   │   ├── api/                          # 📡 API Routes
│   │   │   ├── main.py                   # Main endpoints & middleware
│   │   │   ├── auth.py                   # Authentication routes
│   │   │   └── deps.py                   # Dependency injection
│   │   │
│   │   ├── core/                         # ⚙️ Core Configuration
│   │   │   ├── settings.py               # Environment settings
│   │   │   ├── security.py               # JWT & hashing
│   │   │   ├── jwt.py                    # JWT utilities
│   │   │   ├── agent.py                  # Agent orchestration
│   │   │   ├── graph.py                  # Workflow graph
│   │   │   ├── state.py                  # State definitions
│   │   │   ├── logging.py                # JSON logging
│   │   │   └── rate_limit.py             # Rate limiting
│   │   │
│   │   ├── llm/                          # 🧠 LLM Integration
│   │   │   ├── gemini_client.py          # Google Gemini API
│   │   │   └── client.py                 # LLM client interface
│   │   │
│   │   ├── middleware/
│   │   │   └── request_context.py        # Request ID tracking
│   │   │
│   │   ├── prompts/                      # 📝 Prompt Templates
│   │   │   └── answer_prompt.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── tests/                            # 🧪 Test Suite
│   │   ├── conftest.py                   # Pytest configuration
│   │   ├── test_api_auth.py              # API authentication tests
│   │   ├── test_jwt.py                   # JWT token tests
│   │   ├── test_answer*.py               # Answer agent tests
│   │   ├── test_retriever.py             # Retriever tests
│   │   ├── test_risk.py                  # Risk assessment tests
│   │   ├── test_cross_reference.py       # Conflict detection tests
│   │   ├── test_query_planner.py         # Query planning tests
│   │   ├── test_legal_interpreter.py     # Domain identification tests
│   │   ├── test_rate_limit.py            # Rate limiting tests
│   │   └── test_graph.py                 # Workflow graph tests
│   │
│   ├── .env                              # Environment variables
│   ├── .env.example                      # Environment template
│   ├── requirements.txt                  # Python dependencies
│   ├── Dockerfile                        # Docker image definition
│   ├── .dockerignore                     # Docker ignore file
│   ├── pytest.ini                        # Pytest configuration
│   └── render.yaml                       # Render.com deployment config
│
├── docker-compose.yml                    # Local development compose
├── .gitignore
├── LICENSE
└── README.md                             # This file
```

## 🛠️ Technology Stack

### Frontend (React 19 + TypeScript)
- **React 19**: Latest React framework with concurrent features
- **TypeScript**: Static typing for robust frontend development
- **Vite**: Lightning-fast build tool and dev server (3s startup)
- **React Router v6**: Client-side routing with protection
- **Zustand**: Lightweight state management (authStore, historyStore)
- **Axios**: HTTP client with request/response interceptors
- **Framer Motion**: Smooth animations and transitions
- **Tailwind CSS**: Utility-first CSS framework (dark mode compatible)
- **Lucide React**: 300+ beautiful, consistent icons
- **React Hot Toast**: Non-intrusive toast notifications
- **TanStack React Query**: Server state management
- **date-fns**: Date manipulation and formatting

### Backend (FastAPI + Python)
#### Core Framework
- **FastAPI**: Modern, fast (Starlette-based) API framework
- **Pydantic v2**: Data validation and settings management
- **Uvicorn**: ASGI web server with async support

#### AI & LLM Integration
- **Google Generative AI**: Gemini 2.0 Flash Latest model
- **LangChain**: Prompt management and chain orchestration
- **LangGraph**: Stateful multi-agent workflow orchestration

#### Authentication & Security
- **PyJWT & python-jose**: JWT token generation/verification
- **passlib**: Secure password hashing (bcrypt)
- **CORS Middleware**: Cross-origin resource sharing

#### Application Infrastructure
- **Python-dotenv**: Environment variable management
- **Structured Logging**: JSON-format logging for all operations
- **Request Context**: Unique request ID tracking and tracing

#### Testing & Quality
- **Pytest**: Comprehensive testing framework
- **unittest.mock**: Advanced mocking capabilities
- **TestClient**: FastAPI test client
- **16+ Unit Tests**: 100% critical path coverage

### DevOps & Deployment
- **Docker**: Container images and multi-stage builds
- **Docker Compose**: Local development orchestration
- **Vercel**: Frontend hosting (automatic deployments from Git)
- **Render.com**: Backend hosting (zero-config Python)
- **GitHub Actions**: CI/CD pipeline automation
- **nginx**: Reverse proxy and load balancing (optional)

### Databases & Caching (Optional)
- **PostgreSQL**: Persistent relational database
- **Redis**: In-memory caching and session store
- **FAISS**: Vector similarity search (for future RAG)

## 🚀 Quick Start

### Prerequisites

- **Backend**:
  - Python 3.10 or higher
  - pip package manager
  - Google Gemini API key (free at https://ai.google.dev)
  
- **Frontend**:
  - Node.js 18+ or compatible
  - npm or yarn package manager
  
- **General**:
  - Git
  - Docker (optional, for containerized deployment)

### Complete Local Setup

#### **Step 1: Clone Repository**

```bash
git clone https://github.com/Juniorkz13/ai-portfolio.git
cd ai-portfolio/projects/legal-rag-multi-agent
```

#### **Step 2: Backend Setup**

##### 2.1 Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

##### 2.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

##### 2.3 Configure Environment Variables

```bash
# Create .env file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required backend variables:**
```env
GEMINI_API_KEY=your_google_gemini_api_key
JWT_SECRET_KEY=your_secure_key_min_32_chars  # Generate: openssl rand -hex 16
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=3600
LOG_LEVEL=INFO
DEBUG=false
```

##### 2.4 Run Backend Tests

```bash
pytest -v
# Expected: 16 passed ✓
```

##### 2.5 Start Backend Server

```bash
# Development mode with auto-reload
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend available at: **http://localhost:8000**
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

#### **Step 3: Frontend Setup**

##### 3.1 Install Frontend Dependencies

```bash
cd frontend
npm install
# or yarn install
```

##### 3.2 Configure Frontend Environment

```bash
# Create frontend .env file
cat > .env << EOF
VITE_API_URL=http://localhost:8000
EOF
```

##### 3.3 Start Development Server

```bash
npm run dev
# or yarn dev
```

✅ Frontend available at: **http://localhost:3000**

---

#### **Step 4: Test Complete Flow**

```bash
# In your browser:
1. Navigate to http://localhost:3000
2. Login with: admin@example.com / admin123
3. Ask a legal question
4. Receive comprehensive legal analysis
```

---

### Using Docker (All-in-One)

#### **Option 1: Docker Compose (Recommended)**

```bash
# In project root
docker-compose up -d

# View logs
docker-compose logs -f

# Backend: http://localhost:8000
# Frontend: http://localhost:3000 (if included)
```

#### **Option 2: Manual Docker Build**

```bash
# Build backend image
docker build -t legal-rag-backend:latest .

# Run container
docker run -d \
  --name legal-rag-api \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e JWT_SECRET_KEY=your_secret \
  legal-rag-backend:latest
```

---

### Development Tips

**Frontend + Backend Together (2 Terminals):**

```bash
# Terminal 1: Backend
cd ai-portfolio/projects/legal-rag-multi-agent
source venv/bin/activate
uvicorn app.api.main:app --reload

# Terminal 2: Frontend
cd ai-portfolio/projects/legal-rag-multi-agent/frontend
npm run dev
```

**Rebuild Frontend:**
```bash
cd frontend
npm run build  # Creates dist/ folder
# Deploy dist/ to Vercel or any static hosting
```

**Rebuild Backend:**
```bash
docker build -t legal-rag-backend:latest .
docker push your-registry/legal-rag-backend:latest
```

---

## 🎨 Frontend Features & Components

### User Interface Components

#### **1. Login Page** (`LoginPage.tsx`)
- Modern authentication form with email/password
- JWT token storage in localStorage
- Form validation and error handling
- Beautiful animated entrance effects
- Responsive design for all devices

#### **2. Dashboard Page** (`DashboardPage.tsx`)
- **Main Features:**
  - Textarea for legal question input
  - Real-time analysis with loading states
  - Color-coded risk level badges (Low 🟢 / Medium 🟡 / High 🔴)
  - Detailed analysis results with disclaimers
  - Recommendations and next steps
  - Processing time metrics

- **Sidebar Management:**
  - Recent query history (up to 50 items)
  - Quick access to previous analyses
  - Timestamps and risk levels
  - Clear history functionality
  - Expandable/collapsible sidebar

#### **3. Button Component** (`Button.tsx`)
- Variants: Primary, Secondary, Outline, Ghost
- Sizes: SM, MD, LG
- Loading states with spinner animation
- Accessibility-first design
- Custom className support

### State Management (Zustand)

#### **Auth Store** (`authStore.ts`)
```typescript
// Features:
- User object (username, email)
- JWT token storage
- Authentication state
- Login/logout handlers
- Persistent localStorage sync
- Automatic 401 redirect
```

#### **History Store** (`historyStore.ts`)
```typescript
// Features:
- Query history (max 50 items)
- Complete analysis results
- Timestamps
- Risk levels and domains
- Persistent localStorage
- Quick lookup by ID
```

### API Integration

#### **Axios Client** (`api.ts`)
- **Request Interceptors:**
  - Automatic JWT token injection
  - Content-Type headers
  - 30-second timeout

- **Response Interceptors:**
  - Automatic 401 error handling
  - Token refresh logic
  - Error formatting

- **Endpoints:**
  - `authApi.login()` - JWT authentication
  - `legalApi.analyze()` - Legal analysis requests
  - `legalApi.getModels()` - Available AI models
  - `legalApi.healthCheck()` - Backend health

### Type Definitions (`types/index.ts`)

```typescript
// Main types:
- User (username, email)
- LoginRequest / LoginResponse
- AnalysisRequest / AnalysisResponse
- HistoryItem
- LEGAL_DOMAINS (9 domains with icons)
```

### Styling & Utilities

- **Tailwind CSS**: Dark mode (Dracula-inspired palette)
- **Color System:**
  - Background: #1e2127
  - Cards: #282c34
  - Text: #e6edf3
  - Accent Orange: #d19a66

- **Class Name Utility** (`cn.ts`)
  - Combines clsx + tailwind-merge
  - Conflict resolution for Tailwind classes

---

## 📖 API Documentation

### Base URL

```
http://localhost:8000
```

### Authentication Flow

All endpoints except `/health` and `/api/v1/login` require authentication.

#### Login Endpoint

**Request:**
```bash
POST /api/v1/login
Content-Type: application/json

{
  "username": "admin@example.com",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Test Credentials:**

| Username | Password |
|----------|----------|
| `admin@example.com` | `admin123` |
| `user@example.com` | `password123` |
| `ratelimit_user` | `admin` |

---

#### Legal Analysis Endpoint

**Request:**
```bash
POST /api/v1/analyze
Authorization: Bearer {access_token}
X-API-Key: test-key
Content-Type: application/json

{
  "question": "What are my labor rights if I'm unfairly dismissed?",
  "documents": []
}
```

**Parameters:**
- `question` (required): The legal question or scenario
- `documents` (optional): Array of reference documents or case details

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What are my labor rights if I'm unfairly dismissed?",
  "status": "completed",
  "risk_level": "médio",
  "domain": "Trabalhista",
  "analysis": {
    "answer": "Detailed legal analysis with references to Brazilian Labour Law (CLT)...",
    "disclaimer": "⚠️ IMPORTANT: This is a preliminary legal guidance generated by AI...",
    "summary": "✅ Complete legal analysis regarding your labor rights",
    "documents_processed": 2,
    "queries_generated": 1,
    "has_conflicts": false,
    "is_ambiguous": false,
    "missing_info": [
      "Specific dates of employment",
      "Salary and contract details"
    ],
    "recommendations": [
      "📋 Consult a specialized labor attorney",
      "📄 Gather all employment documentation",
      "⏰ Check applicable legal deadlines"
    ],
    "confidence_score": "75%"
  },
  "agents_used": [
    "query_planner",
    "retriever",
    "cross_reference",
    "legal_interpreter",
    "risk_assessment",
    "answer_agent"
  ],
  "metadata": {
    "workflow_version": "1.0.0",
    "processing_time_ms": 2543,
    "language": "pt-BR"
  }
}
```

**Response Fields:**
- `request_id`: Unique identifier for tracking
- `status`: Either "completed" or "error"
- `risk_level`: 🟢 Low / 🟡 Medium / 🔴 High
- `domain`: Identified legal area (e.g., Trabalhista)
- `analysis.answer`: Main legal guidance from Gemini
- `analysis.disclaimer`: Legal disclaimer
- `analysis.recommendations`: Suggested next steps
- `agents_used`: Which agents processed the request
- `metadata.processing_time_ms`: Response time

---

#### List Available Models Endpoint

**Request:**
```bash
GET /api/v1/models
Authorization: Bearer {access_token}
X-API-Key: test-key
```

**Response:**
```json
{
  "models": [
    "models/gemini-2.0-flash-exp",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash"
  ],
  "current_model": "gemini-2.0-flash-exp",
  "total": 3
}
```

---

#### Health Check Endpoint

**Request:**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "app": "Legal RAG Multi-Agent",
  "version": "1.0.0"
}
```

---

### Response Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Analysis completed successfully |
| 400 | Bad Request | Invalid input format |
| 401 | Unauthorized | Invalid or expired token |
| 429 | Too Many Requests | Rate limit exceeded - wait before retrying |
| 500 | Internal Error | Server error - try again later |

---

## 💻 Usage Examples

### Example 1: Bash/cURL - Analyze Labor Rights

```bash
#!/bin/bash

# Step 1: Login and get token
LOGIN_RESPONSE=$(curl -s -X POST \
  'http://localhost:8000/api/v1/login' \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "admin@example.com",
    "password": "admin123"
  }')

ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

# Step 2: Perform legal analysis
curl -X POST \
  'http://localhost:8000/api/v1/analyze' \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'X-API-Key: test-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Can my employer terminate my contract during sick leave?"
  }' | jq '.'
```

### Example 2: Python

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "test-key"

# Step 1: Authenticate
login_response = requests.post(
    f"{BASE_URL}/api/v1/login",
    json={
        "username": "admin@example.com",
        "password": "admin123"
    }
)

access_token = login_response.json()['access_token']

# Step 2: Analyze question
headers = {
    "Authorization": f"Bearer {access_token}",
    "X-API-Key": API_KEY
}

analysis = requests.post(
    f"{BASE_URL}/api/v1/analyze",
    headers=headers,
    json={"question": "What are my rights regarding overtime payment?"}
)

result = analysis.json()
print(f"Domain: {result['domain']}")
print(f"Risk Level: {result['risk_level']}")
print(f"\n{result['analysis']['answer']}")
```

### Example 3: JavaScript/Node.js

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "test-key";

async function getLegalAdvice(question) {
  // Step 1: Login
  const loginRes = await fetch(`${BASE_URL}/api/v1/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: "admin@example.com",
      password: "admin123"
    })
  });

  const { access_token } = await loginRes.json();

  // Step 2: Analyze
  const analysisRes = await fetch(`${BASE_URL}/api/v1/analyze`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${access_token}`,
      "X-API-Key": API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question })
  });

  return await analysisRes.json();
}

// Usage
getLegalAdvice("What happens if my employer doesn't pay my 13th salary?")
  .then(result => {
    console.log(`Domain: ${result.domain}`);
    console.log(`Risk: ${result.risk_level}`);
    console.log(`\nAdvice:\n${result.analysis.answer}`);
  });
```

## 📚 Supported Legal Domains

The system provides specialized analysis across 9 major Brazilian legal domains:

### 1. 🏢 Labor Law (Direito Trabalhista)

**Focus**: Employee rights, contracts, termination, benefits

**Key Topics**:
- Employment contracts and types (CLT, PJ, apprentice)
- Salary and benefits (13th salary, FGTS, meal vouchers)
- Working hours and overtime
- Vacation and leave policies
- Termination and severance
- Workplace safety and harassment

**Legislation**: CLT (Consolidação das Leis do Trabalho)

---

### 2. 📋 Civil Law (Direito Civil)

**Focus**: Contracts, liability, property, obligations

**Key Topics**:
- Contract formation and interpretation
- Civil liability and damages
- Property rights and obligations
- Inheritance and succession
- Personal relationships
- Tort law

**Legislation**: Civil Code (Código Civil)

---

### 3. 🛍️ Consumer Law (Direito do Consumidor)

**Focus**: Consumer protection, product liability, services

**Key Topics**:
- Product and service defects
- Warranty rights and obligations
- Consumer credit and financing
- Unfair practices and deception
- Right of withdrawal
- Liability of vendors and manufacturers

**Legislation**: CDC (Código de Defesa do Consumidor)

---

### 4. 👨‍👩‍👧 Family Law (Direito da Família)

**Focus**: Marriage, divorce, custody, inheritance

**Key Topics**:
- Marriage and divorce
- Child custody and visitation
- Child support and alimony
- Property division
- Adoption and guardianship
- Paternity and legitimacy

**Legislation**: Family provisions of Civil Code

---

### 5. ⚡ Criminal Law (Direito Penal)

**Focus**: Crime classification, criminal procedure, penalties

**Key Topics**:
- Crime classification and penalties
- Criminal procedure
- Victim rights
- Self-defense and necessity
- Rehabilitation and parole
- Appeal processes

**Legislation**: Criminal Code (Código Penal)

---

### 6. 💰 Tax Law (Direito Tributário)

**Focus**: Income tax, business taxes, fiscal obligations

**Key Topics**:
- Income tax (IR)
- Value-added taxes (ICMS, ISS)
- Corporate taxes
- Tax evasion vs. avoidance
- Tax credits and deductions
- Administrative remedies

**Legislation**: Tax Code (Código Tributário Nacional)

---

### 7. 🏭 Business Law (Direito Empresarial)

**Focus**: Company formation, corporate governance, contracts

**Key Topics**:
- Business structure selection
- Corporate governance and bylaws
- Commercial contracts
- Intellectual property
- Competition law
- Bankruptcy and insolvency

**Legislation**: Business Law Code

---

### 8. 🏦 Social Security Law (Direito Previdenciário)

**Focus**: Retirement, benefits, INSS contributions

**Key Topics**:
- Retirement eligibility and benefits
- Disability and death benefits
- INSS contributions and requirements
- Benefit calculation
- Appeal procedures
- Social assistance

**Legislation**: Social Security Law (Lei nº 8.213/91)

---

### 9. 🏠 Real Estate Law (Direito Imobiliário)

**Focus**: Property, leasing, real estate transactions

**Key Topics**:
- Property ownership and transfer
- Leasing and rental agreements
- Right of way and easements
- Property disputes
- Usucapion (adverse possession)
- Zoning regulations

**Legislation**: Civil Code provisions on real property

---

## 🐳 Deployment

### Backend Deployment

#### Docker Deployment

##### Build Image

```bash
docker build -t legal-rag-multi-agent:latest .
```

##### Run Container

```bash
docker run -d \
  --name legal-rag-api \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  -e JWT_SECRET_KEY=your_secret_key \
  legal-rag-multi-agent:latest
```

##### Docker Compose

```bash
docker-compose up -d
```

#### Cloud Platforms - Backend

##### Render.com (Recommended - Free Tier)

```bash
# render.yaml is already configured
# Just push to GitHub and connect in Render dashboard
# Automatic deployments on push to main branch
```

##### Google Cloud Run

```bash
gcloud run deploy legal-rag-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=your_key,JWT_SECRET_KEY=your_key
```

##### AWS Lambda (Serverless)

```bash
pip install -r requirements.txt -t package/
cd package && zip -r ../deployment.zip . && cd ..
zip deployment.zip -r app/
aws lambda create-function \
  --function-name legal-rag-api \
  --runtime python3.10 \
  --zip-file fileb://deployment.zip
```

##### Heroku

```bash
git push heroku main
```

---

### Frontend Deployment

#### Vercel (Recommended - Zero-Config)

```bash
# Method 1: Connect GitHub repo
# 1. Go to https://vercel.com/new
# 2. Import your GitHub repository
# 3. Select "legal-rag-multi-agent/frontend" as root directory
# 4. Deploy (automatic on every push)

# Method 2: CLI deployment
npm install -g vercel
cd frontend
vercel
```

**Vercel Configuration** (already in `vercel.json`):
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "framework": "vite"
}
```

#### Other Frontend Hosting Options

##### Netlify

```bash
cd frontend
npm run build
# Connect dist/ folder to Netlify
```

##### AWS S3 + CloudFront

```bash
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket-name/
```

##### GitHub Pages

```bash
# Update vite.config.ts
# Uncomment: base: '/legal-rag-multi-agent/'

npm run build
# Push dist/ to gh-pages branch
```

#### Frontend Environment Variables

Create `.env` in `frontend/` directory:
```env
VITE_API_URL=https://your-backend-url.com
# Example for production:
# VITE_API_URL=https://legal-rag-backend-v2.onrender.com
```

---

### Complete Stack Deployment Checklist

#### Backend (Python/FastAPI)
- [ ] Set `DEBUG=false` in production
- [ ] Use strong, unique `JWT_SECRET_KEY` (32+ chars)
- [ ] Enable HTTPS/TLS in reverse proxy
- [ ] Configure CORS for your frontend domain
- [ ] Set up structured logging and monitoring
- [ ] Configure database backups (if using DB)
- [ ] Implement rate limiting at infrastructure level
- [ ] Regular security audits and dependency updates
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configure email notifications

#### Frontend (React/TypeScript)
- [ ] Update `VITE_API_URL` to production backend
- [ ] Enable service worker for PWA
- [ ] Set up analytics (Vercel Analytics, Google Analytics)
- [ ] Configure error boundary
- [ ] Test responsive design on mobile devices
- [ ] Optimize bundle size (check with vite analyze)
- [ ] Enable compression (gzip/brotli)
- [ ] Set security headers (CSP, X-Frame-Options, etc.)
- [ ] Test all authentication flows
- [ ] Monitor Core Web Vitals

#### Full Stack
- [ ] Test end-to-end workflows
- [ ] Load testing (locust, k6)
- [ ] Security penetration testing
- [ ] Document APIs and deployment process
- [ ] Set up automated backups
- [ ] Configure disaster recovery plan

---

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
name: Test and Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - run: pip install -r requirements.txt
    - run: pytest -v
```

---

## 🔗 Integration Examples

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
LEGAL_RAG_URL = "http://localhost:8000"

@app.route("/api/legal-advice", methods=["POST"])
def get_legal_advice():
    question = request.json.get("question")
    # Call Legal RAG API
    response = requests.post(
        f"{LEGAL_RAG_URL}/api/v1/analyze",
        json={"question": question}
    )
    return jsonify(response.json())
```

### Django Integration

```python
# views.py
import requests
from django.http import JsonResponse

def analyze_legal_question(request):
    question = request.POST.get('question')
    response = requests.post(
        'http://localhost:8000/api/v1/analyze',
        json={'question': question}
    )
    return JsonResponse(response.json())
```

---

## 🔜 Roadmap

### Phase 1: Core ✅ **COMPLETE**
- [x] Multi-agent legal system (6 agents)
- [x] JWT authentication with token storage
- [x] Rate limiting (100 req/hour)
- [x] Risk assessment (Low/Medium/High)
- [x] API documentation (Swagger)
- [x] Comprehensive test suite (16 tests)
- [x] Docker containerization
- [x] Health check endpoints
- [x] Structured JSON logging

### Phase 2: Web Frontend ✅ **COMPLETE**
- [x] React 19 + TypeScript dashboard
- [x] Login/authentication pages
- [x] Legal question analysis interface
- [x] Real-time results display
- [x] Risk level color coding
- [x] Query history sidebar
- [x] Zustand state management
- [x] Axios HTTP client with interceptors
- [x] Tailwind CSS dark theme
- [x] Framer Motion animations
- [x] Responsive design (mobile-friendly)
- [x] Toast notifications
- [x] Vercel deployment configuration

### Phase 3: Persistence (In Progress)
- [ ] PostgreSQL integration
- [ ] Extended user management
- [ ] Persistent case/query history
- [ ] Analytics dashboard
- [ ] Session management

### Phase 4: Enhanced AI
- [ ] Vector embeddings (FAISS)
- [ ] Real-time jurisprudence updates
- [ ] Document upload and analysis
- [ ] Multi-language support (ES, EN)
- [ ] Few-shot learning examples
- [ ] Custom knowledge bases

### Phase 5: Mobile & Integration
- [ ] React Native mobile app
- [ ] Email integration
- [ ] Slack bot integration
- [ ] WhatsApp bot
- [ ] Browser extension

### Phase 6: Enterprise
- [ ] Contract analysis module
- [ ] Pay-per-use billing
- [ ] SAML/OAuth authentication
- [ ] Advanced analytics & reporting
- [ ] Custom domain support
- [ ] On-premise deployment options

### Current Status: **Production Ready** 🚀
- ✅ Full-stack implementation complete
- ✅ Frontend and backend integrated
- ✅ Deployable to cloud platforms
- ✅ Ready for user testing

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository: https://github.com/Juniorkz13/ai-portfolio
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
git clone https://github.com/Juniorkz13/ai-portfolio.git
cd ai-portfolio/projects/legal-rag-multi-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest -v
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## ⚠️ Legal Disclaimer

**This system provides preliminary legal guidance ONLY.**

It is **NOT** a substitute for professional legal advice from a qualified attorney.

Always consult with a licensed lawyer for:
- Legal disputes or litigation
- Complex legal questions
- Official legal documents
- Court proceedings
- Critical life decisions

---

## 👨‍💻 About the Author

**José Geraldo do Espirito Santo Júnior**

Full-Stack AI Engineer | Legal Tech Innovator

**LinkedIn**: https://www.linkedin.com/in/josejunior13/

**GitHub**: https://github.com/Juniorkz13

**Portfolio**: https://github.com/Juniorkz13/ai-portfolio

### Expertise

- Advanced AI/ML systems
- Legal domain knowledge
- Cloud-native architecture
- Enterprise software development

### Get in Touch

- **LinkedIn**: https://www.linkedin.com/in/josejunior13/
- **GitHub Issues**: https://github.com/Juniorkz13/ai-portfolio/issues
- **GitHub Discussions**: https://github.com/Juniorkz13/ai-portfolio/discussions

---

## 🙏 Acknowledgments

- Google for Gemini API
- LangChain/LangGraph team
- FastAPI framework
- Open-source community
- Legal domain experts

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Lines of Code | ~3,500 |
| Test Coverage | 100% critical paths |
| API Endpoints | 4 main |
| Legal Domains | 9 |
| AI Agents | 6 |
| Avg Response Time | <3s |

---

## 🎯 Key Takeaways

✅ **Free legal guidance** - No expensive consultation fees  
✅ **24/7 availability** - Anytime, anywhere  
✅ **Expert analysis** - 6 AI agents + Gemini  
✅ **Risk assessment** - Know the complexity  
✅ **Professional recommendations** - Next steps guidance  
✅ **Honest limitations** - Always recommends lawyer  

---

## 📚 Related Projects

Check out other AI projects in the portfolio:

- [AI Portfolio](https://github.com/Juniorkz13/ai-portfolio)
- [Projects Directory](https://github.com/Juniorkz13/ai-portfolio/tree/main/projects)

---

**Built with ❤️ for legal access and digital justice**

_The future of legal guidance is here. Make it yours._