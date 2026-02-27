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

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              API Layer (FastAPI)                       │ │
│  │  • Authentication (JWT)                                │ │
│  │  • Rate Limiting                                       │ │
│  │  • Request Validation                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Multi-Agent Legal Workflow (LangGraph)        │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │  │ Query    │→ │ Retriever│→ │ Cross-   │             │ │
│  │  │ Planner  │  │ Agent    │  │ Ref      │             │ │
│  │  └──────────┘  └──────────┘  └──────────┘             │ │
│  │       ↓              ↓              ↓                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │  │ Legal    │  │ Risk     │  │ Answer   │             │ │
│  │  │ Interpret│→ │ Assessor │→ │ Agent    │             │ │
│  │  └──────────┘  └──────────┘  └──────────┘             │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Gen AI Layer (Google Gemini 2.0)              │ │
│  │  • Natural Language Processing                         │ │
│  │  • Legal Guidance Generation                           │ │
│  │  • Context-Aware Responses                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Response Layer                                 │ │
│  │  • Structured JSON Response                            │ │
│  │  • Risk Assessment                                     │ │
│  │  • Recommendations                                     │ │
│  │  • Legal Disclaimers                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
legal-rag-multi-agent/
├── app/
│   ├── agents/              # Legal Analysis Agents
│   │   ├── query_planner.py
│   │   ├── retriever.py
│   │   ├── cross_reference.py
│   │   ├── legal_interpreter.py
│   │   ├── risk_assessment.py
│   │   └── answer_agent.py
│   │
│   ├── api/                 # FastAPI Application
│   │   ├── main.py          # Main API endpoints
│   │   └── auth.py          # Authentication routes
│   │
│   ├── core/                # Core Functionality
│   │   ├── settings.py      # Configuration management
│   │   ├── security.py      # JWT and authentication
│   │   ├── graph.py         # Multi-agent workflow orchestration
│   │   ├── logging.py       # Structured logging
│   │   └── rate_limit.py    # Rate limiting implementation
│   │
│   └── llm/                 # Language Model Integration
│       └── gemini_client.py # Google Gemini client
│
├── tests/                   # Comprehensive Test Suite
│   ├── test_answer.py
│   ├── test_answer_agent.py
│   ├── test_api_auth.py
│   ├── test_cross_reference.py
│   ├── test_gemini_agent.py
│   ├── test_graph.py
│   ├── test_jwt.py
│   ├── test_legal_interpreter.py
│   ├── test_query_planner.py
│   ├── test_rate_limit.py
│   ├── test_retriever.py
│   └── test_risk.py
│
├── .env                     # Environment variables (create from .env.example)
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker container definition
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## 🛠️ Technology Stack

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and settings management
- **LangGraph**: Orchestration of multi-agent workflows

### Authentication & Security
- **PyJWT**: JWT token generation and verification
- **python-jose**: Secure token handling
- **passlib**: Password hashing and verification

### AI & Machine Learning
- **Google Generative AI**: Gemini 2.0 Flash Latest model
- **LangChain**: LLM orchestration and prompt management

### Application Infrastructure
- **Uvicorn**: ASGI web server
- **Python-dotenv**: Environment variable management
- **Logging**: Native Python structured logging

### Testing & Quality
- **Pytest**: Testing framework
- **unittest.mock**: Mocking and patching
- **Pytest-anyio**: Async test support

### Optional (For Production)
- **PostgreSQL**: Data persistence
- **Redis**: Caching and rate limiting
- **Docker**: Containerization

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Google Gemini API key (free tier available at https://ai.google.dev)
- Git

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/Juniorkz13/ai-portfolio.git
cd ai-portfolio/projects/legal-rag-multi-agent
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

```bash
# Create .env from template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

**Required environment variables:**

```env
# Get API key from: https://ai.google.dev
GEMINI_API_KEY=your_google_gemini_api_key_here

# Generate a secure key (min 32 characters)
# On Linux/macOS: openssl rand -hex 16
# On Windows: python -c "import secrets; print(secrets.token_hex(16))"
JWT_SECRET_KEY=your_secure_secret_key_here_minimum_32_characters

JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=3600

LOG_LEVEL=INFO
APP_NAME=Legal RAG Multi-Agent
APP_VERSION=1.0.0
DEBUG=false
```

#### 5. Run Tests

```bash
pytest -v
```

Expected output:
```
16 passed ✓
```

#### 6. Start the Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: **http://localhost:8000**

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

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

### Docker Deployment

#### Build Image

```bash
docker build -t legal-rag-multi-agent:latest .
```

#### Run Container

```bash
docker run -d \
  --name legal-rag-api \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  -e JWT_SECRET_KEY=your_secret_key \
  legal-rag-multi-agent:latest
```

#### Docker Compose

```bash
docker-compose up -d
```

### Cloud Platforms

#### Google Cloud Run

```bash
gcloud run deploy legal-rag-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=your_key
```

#### AWS Lambda

```bash
pip install -r requirements.txt -t package/
cd package && zip -r ../deployment.zip . && cd ..
zip deployment.zip -r app/
aws lambda create-function \
  --function-name legal-rag-api \
  --runtime python3.10 \
  --zip-file fileb://deployment.zip
```

#### Heroku

```bash
git push heroku main
```

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Use strong `JWT_SECRET_KEY`
- [ ] Enable HTTPS/TLS
- [ ] Configure logging and monitoring
- [ ] Set up database backups
- [ ] Implement rate limiting at infrastructure level
- [ ] Regular security audits

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

### Phase 1: Core ✅
- [x] Multi-agent legal system
- [x] JWT authentication
- [x] Rate limiting
- [x] Risk assessment
- [x] API documentation

### Phase 2: Persistence (In Progress)
- [ ] PostgreSQL integration
- [ ] User management
- [ ] Case history
- [ ] Analytics dashboard

### Phase 3: Enhanced AI
- [ ] Vector embeddings
- [ ] Real-time jurisprudence
- [ ] Document upload
- [ ] Multi-language support

### Phase 4: Web & Mobile
- [ ] React dashboard
- [ ] Mobile app
- [ ] Email integration
- [ ] Slack/WhatsApp bots

### Phase 5: Enterprise
- [ ] Contract analysis
- [ ] Pay-per-use billing
- [ ] SAML authentication
- [ ] Advanced analytics

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