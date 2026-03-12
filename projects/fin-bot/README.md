# 🤖 FinBot --- AI Financial Assistant

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-framework-green)
![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> An AI-powered financial assistant that combines **natural language
> processing**, a **Telegram bot**, and a **web dashboard** to help
> users track and understand their personal finances.

FinBot allows users to register **expenses and income using natural
language**, view **financial insights**, analyze **spending patterns**,
and manage their finances through both **Telegram** and a **web
interface**.

------------------------------------------------------------------------

# ✨ Features

### 💬 Natural Language Transactions

Register financial transactions simply by typing messages like:

    I spent 25 on lunch today
    Received 2500 salary
    Spent 15 on Uber yesterday

The AI extracts:

-   transaction type
-   amount
-   category
-   description
-   date

------------------------------------------------------------------------

### 📊 Financial Insights

FinBot automatically generates insights such as:

-   largest expense category
-   income vs expenses comparison
-   monthly financial balance
-   spending patterns

Example insight:

> 💡 *Your largest expense this month was Food (R\$350).*

------------------------------------------------------------------------

### 📈 Web Dashboard

The web dashboard allows users to:

-   visualize spending with charts
-   view recent transactions
-   analyze monthly summaries
-   explore financial insights

------------------------------------------------------------------------

### 📂 CSV Import

Users can upload CSV files to import financial data.

Useful for:

-   bank exports
-   historical data imports
-   bulk transaction upload

------------------------------------------------------------------------

### 🤖 Telegram Bot

FinBot integrates with Telegram allowing users to interact with their
finances directly in chat.

Examples:

    Spent 35 on lunch today
    Received 1500 salary
    What did I spend this month?
    Show my recent transactions
    Give me a financial insight

------------------------------------------------------------------------

### 👥 Multi‑User System

The platform supports multiple users with full data isolation.

Each user has:

-   personal transactions
-   independent financial insights
-   private dashboard data

Users can be created via:

-   Web interface
-   Telegram interaction

------------------------------------------------------------------------

# 🏗 Architecture

Below is a simplified architecture of FinBot.

                        ┌─────────────────┐
                        │    Telegram     │
                        │      Users      │
                        └────────┬────────┘
                                 │
                                 ▼
                         ┌─────────────┐
                         │ Telegram Bot│
                         │   Webhook   │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   FastAPI   │
                         │   Backend   │
                         └──────┬──────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
      ┌───────────┐      ┌────────────┐       ┌─────────────┐
      │ AI Agents │      │  Services  │       │   Tools     │
      │ (LangChain)│     │ Business   │       │ Utilities   │
      └─────┬─────┘      │ Logic      │       └──────┬──────┘
            │             └──────┬─────┘              │
            └──────────────┬─────┴──────────────┬─────┘
                           ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ PostgreSQL   │     │   Frontend   │
                     │ Database     │     │ React / Vite │
                     └──────────────┘     └──────────────┘

------------------------------------------------------------------------

# 🧠 AI Capabilities

FinBot uses AI agents to:

-   parse natural language transactions
-   categorize financial entries
-   generate financial insights
-   analyze spending patterns

The project demonstrates how **AI workflows can orchestrate tools and
services in real applications**.

------------------------------------------------------------------------

# 🧪 Testing

The project includes **integration tests** validating:

-   user creation
-   transaction registration
-   financial summaries
-   insights generation
-   CSV imports
-   multi-user data isolation

Run tests with:

    pytest -v

------------------------------------------------------------------------

# 🚀 Running Locally

## Clone repository

    git clone https://github.com/Juniorkz13/fin-bot.git
    cd fin-bot

## Create environment

    python -m venv .venv
    source .venv/bin/activate

## Install dependencies

    pip install -e ".[dev]"

## Configure environment

Create `.env` based on:

    .env.example

Example:

    DATABASE_URL=postgresql://...
    TELEGRAM_BOT_TOKEN=...
    OPENAI_API_KEY=...

## Run backend

    uvicorn backend.main:app --reload

API documentation:

    http://localhost:8000/docs

## Run frontend

    cd frontend
    npm install
    npm run dev

------------------------------------------------------------------------

# 📌 Example Workflow

1️⃣ User sends message via Telegram

    Spent 20 on coffee

2️⃣ AI extracts transaction

3️⃣ Backend stores transaction in PostgreSQL

4️⃣ Dashboard updates

5️⃣ Insights are generated

------------------------------------------------------------------------

# 🧩 Tech Stack

### Backend

-   FastAPI
-   SQLAlchemy
-   PostgreSQL
-   Pydantic

### AI

-   LangChain
-   AI Agents
-   Natural Language Parsing

### Frontend

-   React
-   Vite
-   Chart libraries

### Bot

-   Telegram Bot API

### Testing

-   Pytest

------------------------------------------------------------------------

# 🧭 Key Engineering Decisions

### 1️⃣ AI Agents + Tool Architecture

The system separates:

-   **Agents** → reasoning and interpretation
-   **Services** → business logic
-   **Tools** → reusable operations

This keeps the AI layer modular and testable.

------------------------------------------------------------------------

### 2️⃣ Multi‑User Data Isolation

All queries are scoped by `user_id` ensuring:

-   privacy
-   correct insights per user
-   safe concurrent usage

------------------------------------------------------------------------

### 3️⃣ Telegram + Web Dual Interface

The system was designed with **two entry points**:

-   conversational interface (Telegram)
-   visual interface (dashboard)

Both interact with the same API layer.

------------------------------------------------------------------------

### 4️⃣ Service Layer Pattern

Business logic lives in **services**, not API routes.

Benefits:

-   easier testing
-   better maintainability
-   reusable workflows

------------------------------------------------------------------------

# 🔮 Possible Future Improvements

-   bank API integrations
-   automatic categorization with ML
-   budgeting system
-   financial forecasting
-   mobile notifications

------------------------------------------------------------------------

# 👨‍💻 Author

**José Geraldo do Espirito Santo Júnior**\
🇧🇷 Brazil

🔗 LinkedIn\
https://www.linkedin.com/in/josejunior13/

------------------------------------------------------------------------

⭐ If you found this project interesting, consider giving it a star!
