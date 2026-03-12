const userNameInput = document.getElementById("user-name");
const userIdInput = document.getElementById("user-id");
const telegramIdInput = document.getElementById("telegram-id");
const createUserButton = document.getElementById("create-user");
const saveUserIdButton = document.getElementById("save-user-id");
const resolveTelegramUserButton = document.getElementById("resolve-telegram-user");
const generateLinkCodeButton = document.getElementById("generate-link-code");
const currentUserStatus = document.getElementById("current-user-status");
const telegramLinkBox = document.getElementById("telegram-link-box");
const summaryMonth = document.getElementById("summary-month");
const summaryYear = document.getElementById("summary-year");
const refreshSummaryButton = document.getElementById("refresh-summary");
const summaryIncome = document.getElementById("summary-income");
const summaryExpenses = document.getElementById("summary-expenses");
const summaryBalance = document.getElementById("summary-balance");
const summaryStatus = document.getElementById("summary-status");
const categoryList = document.getElementById("category-list");
const insightsList = document.getElementById("insights-list");
const balanceChart = document.getElementById("balance-chart");
const categoryChart = document.getElementById("category-chart");
const transactionForm = document.getElementById("transaction-form");
const transactionType = document.getElementById("transaction-type");
const transactionAmount = document.getElementById("transaction-amount");
const transactionCategory = document.getElementById("transaction-category");
const transactionDescription = document.getElementById("transaction-description");
const transactionDate = document.getElementById("transaction-date");
const categoryOptions = document.getElementById("category-options");
const formStatus = document.getElementById("form-status");
const csvImportForm = document.getElementById("csv-import-form");
const csvFileInput = document.getElementById("csv-file");
const csvImportStatus = document.getElementById("csv-import-status");
const csvImportResults = document.getElementById("csv-import-results");
const transactionList = document.getElementById("transaction-list");
const transactionsStatus = document.getElementById("transactions-status");
const recentLimit = document.getElementById("recent-limit");
const refreshTransactionsButton = document.getElementById("refresh-transactions");

const USER_ID_STORAGE_KEY = "finbot-user-id";
const TELEGRAM_ID_STORAGE_KEY = "finbot-telegram-id";
const USER_NAME_STORAGE_KEY = "finbot-user-name";

bootstrap();

function bootstrap() {
  const today = new Date();
  transactionDate.value = today.toISOString().slice(0, 10);
  populateMonthOptions(today);
  populateYearOptions(today);
  userNameInput.value = localStorage.getItem(USER_NAME_STORAGE_KEY) || "";
  userIdInput.value = localStorage.getItem(USER_ID_STORAGE_KEY) || "";
  telegramIdInput.value = localStorage.getItem(TELEGRAM_ID_STORAGE_KEY) || "";
  bindEvents();
  refreshCurrentUser();
  if (hasUserIdentity()) {
    loadCategories();
    loadSummary();
    loadTransactions();
  }
}

function bindEvents() {
  createUserButton.addEventListener("click", createManualUser);
  saveUserIdButton.addEventListener("click", () => {
    persistUserId();
    refreshCurrentUser();
    if (hasUserIdentity()) {
      loadSummary();
      loadTransactions();
    }
  });
  resolveTelegramUserButton.addEventListener("click", resolveUserFromTelegramId);
  generateLinkCodeButton.addEventListener("click", generateTelegramLinkCode);

  refreshSummaryButton.addEventListener("click", loadSummary);
  refreshTransactionsButton.addEventListener("click", loadTransactions);
  transactionType.addEventListener("change", loadCategories);
  transactionForm.addEventListener("submit", handleTransactionSubmit);
  csvImportForm.addEventListener("submit", handleCsvImportSubmit);
}

function populateMonthOptions(today) {
  const formatter = new Intl.DateTimeFormat("pt-BR", { month: "long" });
  for (let month = 1; month <= 12; month += 1) {
    const option = document.createElement("option");
    option.value = String(month);
    option.textContent = formatter.format(new Date(today.getFullYear(), month - 1, 1));
    if (month === today.getMonth() + 1) {
      option.selected = true;
    }
    summaryMonth.append(option);
  }
}

function populateYearOptions(today) {
  const currentYear = today.getFullYear();
  for (let year = currentYear - 1; year <= currentYear + 1; year += 1) {
    const option = document.createElement("option");
    option.value = String(year);
    option.textContent = String(year);
    if (year === currentYear) {
      option.selected = true;
    }
    summaryYear.append(option);
  }
}

function persistUserId() {
  localStorage.setItem(USER_NAME_STORAGE_KEY, userNameInput.value.trim());
  localStorage.setItem(USER_ID_STORAGE_KEY, userIdInput.value.trim());
  localStorage.setItem(TELEGRAM_ID_STORAGE_KEY, telegramIdInput.value.trim());
}

function hasUserIdentity() {
  return Boolean(userIdInput.value.trim() || telegramIdInput.value.trim());
}

function getUserId() {
  const value = userIdInput.value.trim();
  if (value) {
    localStorage.setItem(USER_ID_STORAGE_KEY, value);
  }
  return value;
}

function getHeaders() {
  return {
    "Content-Type": "application/json",
    ...getAuthHeaders(),
  };
}

function getAuthHeaders() {
  const userId = getUserId();
  const telegramId = telegramIdInput.value.trim();

  if (userId) {
    return {
      "X-User-Id": userId,
    };
  }

  if (telegramId) {
    localStorage.setItem(TELEGRAM_ID_STORAGE_KEY, telegramId);
    return {
      "X-Telegram-Id": telegramId,
    };
  }

  if (!userId) {
    throw new Error("Informe o X-User-Id ou o Telegram ID antes de consultar a API.");
  }
  return {};
}

async function loadCategories() {
  const type = transactionType.value;
  try {
    const response = await fetch(`/api/categories?type=${encodeURIComponent(type)}`);
    if (!response.ok) {
      throw new Error("Nao foi possivel carregar categorias.");
    }
    const categories = await response.json();
    categoryOptions.innerHTML = "";
    categories.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.name;
      categoryOptions.append(option);
    });
  } catch (error) {
    categoryOptions.innerHTML = "";
  }
}

async function loadSummary() {
  summaryStatus.textContent = "Carregando resumo...";
  categoryList.innerHTML = "";
  insightsList.innerHTML = "";
  balanceChart.innerHTML = "";
  categoryChart.innerHTML = "";

  try {
    const response = await fetch(
      `/api/summary/month?month=${summaryMonth.value}&year=${summaryYear.value}`,
      { headers: getHeaders() }
    );
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const summary = await response.json();
    summaryIncome.textContent = formatCurrency(summary.total_income);
    summaryExpenses.textContent = formatCurrency(summary.total_expenses);
    summaryBalance.textContent = formatCurrency(summary.balance);
    renderCategories(summary.expenses_by_category);
    renderInsights(summary.insights || []);
    renderBalanceChart(summary);
    renderCategoryChart(summary.expenses_by_category);
    summaryStatus.textContent = "Resumo atualizado.";
  } catch (error) {
    summaryIncome.textContent = "R$ 0,00";
    summaryExpenses.textContent = "R$ 0,00";
    summaryBalance.textContent = "R$ 0,00";
    categoryList.innerHTML = `<li class="empty-state">${error.message}</li>`;
    insightsList.innerHTML = `<li class="empty-state">Sem insights disponiveis.</li>`;
    balanceChart.innerHTML = `<p class="empty-state">Sem dados para o grafico.</p>`;
    categoryChart.innerHTML = `<p class="empty-state">Sem dados para o grafico.</p>`;
    summaryStatus.textContent = "Nao foi possivel carregar o resumo.";
  }
}

async function refreshCurrentUser() {
  currentUserStatus.textContent = "";
  telegramLinkBox.innerHTML = "";

  if (!hasUserIdentity()) {
    currentUserStatus.textContent = "Crie um usuário web ou informe um identificador para começar.";
    return;
  }

  try {
    const response = await fetch("/api/me", {
      headers: getAuthHeaders(),
    });
    if (!response.ok) {
      return;
    }
    const user = await response.json();
    userIdInput.value = user.id;
    userNameInput.value = user.name || "";
    telegramIdInput.value = user.telegram_id ? String(user.telegram_id) : "";
    persistUserId();
    currentUserStatus.textContent = buildCurrentUserLabel(user);
  } catch (error) {
    currentUserStatus.textContent = "";
  }
}

async function resolveUserFromTelegramId() {
  const telegramId = telegramIdInput.value.trim();
  currentUserStatus.textContent = "Resolvendo usuario...";

  if (!telegramId) {
    currentUserStatus.textContent = "Informe um Telegram ID.";
    return;
  }

  try {
    const response = await fetch(`/api/users/by-telegram/${encodeURIComponent(telegramId)}`);
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const user = await response.json();
    userIdInput.value = user.id;
    userNameInput.value = user.name || "";
    telegramIdInput.value = user.telegram_id ? String(user.telegram_id) : "";
    persistUserId();
    currentUserStatus.textContent = buildCurrentUserLabel(user);
    loadCategories();
    loadSummary();
    loadTransactions();
  } catch (error) {
    currentUserStatus.textContent = error.message;
  }
}

async function createManualUser() {
  const name = userNameInput.value.trim();
  currentUserStatus.textContent = "Criando usuário...";

  if (!name) {
    currentUserStatus.textContent = "Informe um nome para criar o usuário.";
    return;
  }

  try {
    const response = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const user = await response.json();
    userIdInput.value = user.id;
    telegramIdInput.value = user.telegram_id ? String(user.telegram_id) : "";
    userNameInput.value = user.name || "";
    persistUserId();
    currentUserStatus.textContent = buildCurrentUserLabel(user);
    loadCategories();
    loadSummary();
    loadTransactions();
  } catch (error) {
    currentUserStatus.textContent = error.message;
  }
}

async function generateTelegramLinkCode() {
  telegramLinkBox.innerHTML = "";
  currentUserStatus.textContent = "Gerando código de vínculo...";

  if (!userIdInput.value.trim()) {
    currentUserStatus.textContent = "Crie ou selecione um usuário web antes de gerar o código.";
    return;
  }

  try {
    const response = await fetch("/api/me/telegram-link-code", {
      method: "POST",
      headers: getAuthHeaders(),
    });
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const payload = await response.json();
    currentUserStatus.textContent = "Código gerado com sucesso.";
    telegramLinkBox.innerHTML = `
      <strong>Código: ${escapeHtml(payload.code)}</strong>
      <span class="transaction-meta">Expira em: ${escapeHtml(payload.expires_at)}</span>
      <span class="transaction-meta">No Telegram, envie: /link ${escapeHtml(payload.code)}</span>
    `;
  } catch (error) {
    currentUserStatus.textContent = error.message;
  }
}

function buildCurrentUserLabel(user) {
  if (user.telegram_id) {
    return `Usuario ativo: ${user.name} (${user.telegram_id})`;
  }
  return `Usuario ativo: ${user.name}`;
}

function renderCategories(items) {
  if (!items.length) {
    categoryList.innerHTML = `<li class="empty-state">Sem despesas por categoria neste periodo.</li>`;
    return;
  }

  categoryList.innerHTML = items
    .map(
      (item) => `
        <li>
          <div class="category-name">
            <strong>${escapeHtml(item.category)}</strong>
          </div>
          <strong>${formatCurrency(item.total)}</strong>
        </li>
      `
    )
    .join("");
}

function renderInsights(items) {
  if (!items.length) {
    insightsList.innerHTML = `<li class="empty-state">Sem insights para este periodo.</li>`;
    return;
  }

  insightsList.innerHTML = items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function renderBalanceChart(summary) {
  const income = Number(summary.total_income) || 0;
  const expenses = Number(summary.total_expenses) || 0;
  const total = Math.max(income + expenses, 1);
  const incomeWidth = (income / total) * 100;
  const expenseWidth = (expenses / total) * 100;

  balanceChart.innerHTML = `
    <div class="stack-track">
      <div class="stack-segment income-segment" style="width:${incomeWidth}%"></div>
      <div class="stack-segment expense-segment" style="width:${expenseWidth}%"></div>
    </div>
    <div class="stack-values">
      <strong>${formatCurrency(income)}</strong>
      <strong>${formatCurrency(expenses)}</strong>
    </div>
  `;
}

function renderCategoryChart(items) {
  if (!items.length) {
    categoryChart.innerHTML = `<p class="empty-state">Sem despesas por categoria neste periodo.</p>`;
    return;
  }

  const maxValue = Math.max(...items.map((item) => Number(item.total) || 0), 1);
  categoryChart.innerHTML = items
    .map((item) => {
      const value = Number(item.total) || 0;
      const width = (value / maxValue) * 100;
      return `
        <div class="bar-row">
          <div class="bar-meta">
            <span>${escapeHtml(item.category)}</span>
            <strong>${formatCurrency(value)}</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${width}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

async function loadTransactions() {
  transactionsStatus.textContent = "Carregando transacoes...";
  transactionList.innerHTML = "";

  try {
    const response = await fetch(`/api/transactions?limit=${recentLimit.value}`, {
      headers: getHeaders(),
    });
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const payload = await response.json();
    renderTransactions(payload.items);
    transactionsStatus.textContent = "Lista atualizada.";
  } catch (error) {
    transactionList.innerHTML = `<li class="empty-state">${error.message}</li>`;
    transactionsStatus.textContent = "Nao foi possivel carregar as transacoes.";
  }
}

function renderTransactions(items) {
  if (!items.length) {
    transactionList.innerHTML = `<li class="empty-state">Nenhuma transacao recente encontrada.</li>`;
    return;
  }

  transactionList.innerHTML = items
    .map(
      (item) => `
        <li>
          <div class="transaction-main">
            <strong>${escapeHtml(item.description || item.category)}</strong>
            <span class="transaction-meta">${item.date} • ${escapeHtml(item.category)}</span>
          </div>
          <div class="transaction-main" style="align-items:flex-end;">
            <span class="transaction-tag ${item.type}">${item.type === "income" ? "Receita" : "Despesa"}</span>
            <strong>${formatCurrency(item.amount)}</strong>
          </div>
        </li>
      `
    )
    .join("");
}

async function handleTransactionSubmit(event) {
  event.preventDefault();
  formStatus.textContent = "Salvando...";

  const payload = {
    type: transactionType.value,
    amount: transactionAmount.value,
    category: transactionCategory.value.trim(),
    description: transactionDescription.value.trim() || null,
    date: transactionDate.value,
  };

  try {
    const response = await fetch("/api/transactions", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    await response.json();
    formStatus.textContent = "Transacao criada com sucesso.";
    transactionAmount.value = "";
    transactionCategory.value = "";
    transactionDescription.value = "";
    loadSummary();
    loadTransactions();
  } catch (error) {
    formStatus.textContent = error.message;
  }
}

async function handleCsvImportSubmit(event) {
  event.preventDefault();
  csvImportStatus.textContent = "Importando...";
  csvImportResults.innerHTML = "";

  const file = csvFileInput.files && csvFileInput.files[0];
  if (!file) {
    csvImportStatus.textContent = "Selecione um arquivo CSV.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/transactions/import", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });
    if (!response.ok) {
      const detail = await extractError(response);
      throw new Error(detail);
    }
    const payload = await response.json();
    csvImportStatus.textContent = `Importacao concluida: ${payload.imported_count} importadas, ${payload.skipped_count} ignoradas.`;
    renderImportResults(payload);
    csvImportForm.reset();
    loadSummary();
    loadTransactions();
  } catch (error) {
    csvImportStatus.textContent = error.message;
  }
}

function renderImportResults(payload) {
  const lines = [
    `<li>${payload.imported_count} transacao(oes) importada(s) com sucesso.</li>`,
  ];

  if (payload.errors && payload.errors.length) {
    payload.errors.forEach((item) => {
      lines.push(`<li>Linha ${item.line_number}: ${escapeHtml(item.error)}</li>`);
    });
  }

  csvImportResults.innerHTML = lines.join("");
}

async function extractError(response) {
  try {
    const payload = await response.json();
    return payload.detail || "Ocorreu um erro ao consultar a API.";
  } catch {
    return "Ocorreu um erro ao consultar a API.";
  }
}

function formatCurrency(value) {
  const number = Number(value);
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(Number.isFinite(number) ? number : 0);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
