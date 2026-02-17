import { useEffect, useMemo, useState } from "react";

const DEFAULT_FORM = {
  tenure: "",
  monthly_charges: "",
  total_charges: "",
  contract_type: "Month-to-month",
  payment_method: "Electronic check",
  internet_service: "Fiber optic"
};

const CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"];
const PAYMENT_METHODS = [
  "Electronic check",
  "Mailed check",
  "Bank transfer (automatic)",
  "Credit card (automatic)"
];
const INTERNET_SERVICES = ["Fiber optic", "DSL", "No"];

function getApiBase() {
  return import.meta.env.VITE_API_BASE || "http://localhost:8000";
}

function parseNumber(value) {
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function mapPayload(form) {
  return {
    tenure: Number(form.tenure),
    monthly_charges: Number(form.monthly_charges),
    total_charges: Number(form.total_charges),
    contract_type: form.contract_type,
    payment_method: form.payment_method,
    internet_service: form.internet_service
  };
}

function formatProbability(probability) {
  return `${(probability * 100).toFixed(2)}%`;
}

export default function App() {
  const apiBase = useMemo(() => getApiBase(), []);
  const docsUrl = useMemo(() => `${getApiBase()}/docs`, []);
  const [status, setStatus] = useState({ loading: true, ok: false });
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(() => {
    const stored = localStorage.getItem("churn_history");
    return stored ? JSON.parse(stored) : [];
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(null);

  useEffect(() => {
    localStorage.setItem("churn_history", JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    let isMounted = true;
    setStatus({ loading: true, ok: false });
    fetch(`${apiBase}/`)
      .then((res) => {
        if (!res.ok) throw new Error("API indisponível");
        return res.json();
      })
      .then(() => {
        if (isMounted) setStatus({ loading: false, ok: true });
      })
      .catch(() => {
        if (isMounted) setStatus({ loading: false, ok: false });
      });

    return () => {
      isMounted = false;
    };
  }, [apiBase]);

  const onChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const validateForm = (payload) => {
    if (payload.tenure == null || payload.tenure < 0) return "Tenure inválido";
    if (payload.monthly_charges == null || payload.monthly_charges < 0) {
      return "Monthly charges inválido";
    }
    if (payload.total_charges == null || payload.total_charges < 0) {
      return "Total charges inválido";
    }
    return "";
  };

  const submitPrediction = async (payload) => {
    const response = await fetch(`${apiBase}/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error("Falha ao obter predição");
    }

    return response.json();
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);

    const payload = mapPayload(form);
    const validation = validateForm({
      ...payload,
      tenure: parseNumber(form.tenure),
      monthly_charges: parseNumber(form.monthly_charges),
      total_charges: parseNumber(form.total_charges)
    });

    if (validation) {
      setError(validation);
      setLoading(false);
      return;
    }

    try {
      const prediction = await submitPrediction(payload);
      const record = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        input: payload,
        output: prediction
      };
      setResult(record);
      setHistory((prev) => [record, ...prev]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (record) => {
    setEditing({
      ...record,
      form: {
        tenure: String(record.input.tenure),
        monthly_charges: String(record.input.monthly_charges),
        total_charges: String(record.input.total_charges),
        contract_type: record.input.contract_type,
        payment_method: record.input.payment_method,
        internet_service: record.input.internet_service
      }
    });
  };

  const handleUpdate = async (event) => {
    event.preventDefault();
    if (!editing) return;
    setError("");
    setLoading(true);

    const payload = mapPayload(editing.form);
    const validation = validateForm({
      ...payload,
      tenure: parseNumber(editing.form.tenure),
      monthly_charges: parseNumber(editing.form.monthly_charges),
      total_charges: parseNumber(editing.form.total_charges)
    });

    if (validation) {
      setError(validation);
      setLoading(false);
      return;
    }

    try {
      const prediction = await submitPrediction(payload);
      const updated = {
        ...editing,
        input: payload,
        output: prediction,
        updatedAt: new Date().toISOString()
      };
      setHistory((prev) =>
        prev.map((item) => (item.id === updated.id ? updated : item))
      );
      setEditing(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (id) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleClear = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen">
      <header className="bg-white border-b border-slate-200">
        <div className="mx-auto max-w-6xl px-6 py-6 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold text-brand-600">MLOps</p>
            <h1 className="text-2xl font-semibold">Customer Churn Prediction</h1>
          </div>
          <div className="flex gap-3">
            <a
              href="/"
              className="text-sm text-slate-500 hover:text-slate-700"
            >
              Dashboard
            </a>
            <a
              href={docsUrl}
              className="text-sm text-slate-500 hover:text-slate-700"
            >
              API Docs
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8 space-y-8">
        <section className="grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-sm text-slate-500">Status da API</p>
            <div className="mt-2 flex items-center gap-3">
              <span
                className={`h-3 w-3 rounded-full ${
                  status.loading
                    ? "bg-slate-300"
                    : status.ok
                    ? "bg-emerald-500"
                    : "bg-rose-500"
                }`}
              />
              <p className="text-base font-semibold">
                {status.loading
                  ? "Verificando..."
                  : status.ok
                  ? "Online"
                  : "Offline"}
              </p>
            </div>
            <p className="mt-4 text-xs text-slate-400">Base: {apiBase}</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-sm text-slate-500">Predições na sessão</p>
            <p className="mt-2 text-3xl font-semibold text-slate-900">
              {history.length}
            </p>
            <p className="mt-2 text-xs text-slate-400">
              Salvas localmente no navegador
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-sm text-slate-500">Última predição</p>
            {result ? (
              <div className="mt-2 space-y-1">
                <p className="text-2xl font-semibold text-slate-900">
                  {formatProbability(result.output.churn_probability)}
                </p>
                <p
                  className={`text-sm font-semibold ${
                    result.output.churn_prediction === 1
                      ? "text-rose-600"
                      : "text-emerald-600"
                  }`}
                >
                  {result.output.churn_prediction === 1
                    ? "Risco de churn"
                    : "Cliente retido"}
                </p>
              </div>
            ) : (
              <p className="mt-2 text-sm text-slate-400">
                Nenhuma predição ainda
              </p>
            )}
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">Nova predição</h2>
                <p className="text-sm text-slate-500">
                  Preencha os dados do cliente para estimar churn.
                </p>
              </div>
            </div>

            <form className="mt-6 grid gap-4" onSubmit={handleSubmit}>
              <div className="grid gap-4 md:grid-cols-3">
                <label className="text-sm font-medium text-slate-700">
                  Tenure (meses)
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="tenure"
                    type="number"
                    min="0"
                    value={form.tenure}
                    onChange={onChange}
                    required
                  />
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Monthly charges
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="monthly_charges"
                    type="number"
                    min="0"
                    step="0.01"
                    value={form.monthly_charges}
                    onChange={onChange}
                    required
                  />
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Total charges
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="total_charges"
                    type="number"
                    min="0"
                    step="0.01"
                    value={form.total_charges}
                    onChange={onChange}
                    required
                  />
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <label className="text-sm font-medium text-slate-700">
                  Contract type
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="contract_type"
                    value={form.contract_type}
                    onChange={onChange}
                  >
                    {CONTRACT_TYPES.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Payment method
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="payment_method"
                    value={form.payment_method}
                    onChange={onChange}
                  >
                    {PAYMENT_METHODS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Internet service
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="internet_service"
                    value={form.internet_service}
                    onChange={onChange}
                  >
                    {INTERNET_SERVICES.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {error ? (
                <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">
                  {error}
                </div>
              ) : null}

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loading ? "Processando..." : "Gerar predição"}
                </button>
                <button
                  type="button"
                  onClick={() => setForm(DEFAULT_FORM)}
                  className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50"
                >
                  Limpar
                </button>
              </div>
            </form>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Resultado</h2>
            {result ? (
              <div className="mt-4 space-y-3">
                <p className="text-sm text-slate-500">Probabilidade de churn</p>
                <p className="text-3xl font-semibold text-slate-900">
                  {formatProbability(result.output.churn_probability)}
                </p>
                <div
                  className={`rounded-lg px-3 py-2 text-sm font-semibold ${
                    result.output.churn_prediction === 1
                      ? "bg-rose-50 text-rose-700"
                      : "bg-emerald-50 text-emerald-700"
                  }`}
                >
                  {result.output.churn_prediction === 1
                    ? "Alto risco de churn"
                    : "Baixo risco de churn"}
                </div>
                <p className="text-xs text-slate-400">
                  Gerado em {new Date(result.createdAt).toLocaleString("pt-BR")}
                </p>
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-400">
                Nenhuma predição disponível.
              </p>
            )}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Histórico</h2>
              <p className="text-sm text-slate-500">
                CRUD completo sobre as predições salvas no navegador.
              </p>
            </div>
            <button
              type="button"
              onClick={handleClear}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50"
            >
              Limpar histórico
            </button>
          </div>

          {history.length === 0 ? (
            <p className="mt-4 text-sm text-slate-400">
              Nenhuma predição salva ainda.
            </p>
          ) : (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-400">
                  <tr>
                    <th className="py-3">Cliente</th>
                    <th className="py-3">Probabilidade</th>
                    <th className="py-3">Predição</th>
                    <th className="py-3">Ações</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {history.map((item) => (
                    <tr key={item.id} className="text-slate-700">
                      <td className="py-4">
                        <p className="font-semibold">
                          {item.input.contract_type} · {item.input.payment_method}
                        </p>
                        <p className="text-xs text-slate-400">
                          Tenure {item.input.tenure} · Charges {item.input.monthly_charges}
                        </p>
                      </td>
                      <td className="py-4">
                        {formatProbability(item.output.churn_probability)}
                      </td>
                      <td className="py-4">
                        {item.output.churn_prediction === 1
                          ? "Churn"
                          : "Retenção"}
                      </td>
                      <td className="py-4">
                        <div className="flex flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => handleEdit(item)}
                            className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                          >
                            Editar
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDelete(item.id)}
                            className="rounded-lg border border-rose-200 px-3 py-1 text-xs font-semibold text-rose-600 hover:bg-rose-50"
                          >
                            Remover
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      {editing ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
          <div className="w-full max-w-2xl rounded-2xl bg-white p-6 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Editar predição</h3>
              <button
                type="button"
                onClick={() => setEditing(null)}
                className="text-sm text-slate-500 hover:text-slate-700"
              >
                Fechar
              </button>
            </div>
            <form className="mt-4 grid gap-4" onSubmit={handleUpdate}>
              <div className="grid gap-4 md:grid-cols-3">
                <label className="text-sm font-medium text-slate-700">
                  Tenure (meses)
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="tenure"
                    type="number"
                    min="0"
                    value={editing.form.tenure}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: { ...prev.form, tenure: event.target.value }
                      }))
                    }
                    required
                  />
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Monthly charges
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="monthly_charges"
                    type="number"
                    min="0"
                    step="0.01"
                    value={editing.form.monthly_charges}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: {
                          ...prev.form,
                          monthly_charges: event.target.value
                        }
                      }))
                    }
                    required
                  />
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Total charges
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="total_charges"
                    type="number"
                    min="0"
                    step="0.01"
                    value={editing.form.total_charges}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: {
                          ...prev.form,
                          total_charges: event.target.value
                        }
                      }))
                    }
                    required
                  />
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <label className="text-sm font-medium text-slate-700">
                  Contract type
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="contract_type"
                    value={editing.form.contract_type}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: {
                          ...prev.form,
                          contract_type: event.target.value
                        }
                      }))
                    }
                  >
                    {CONTRACT_TYPES.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Payment method
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="payment_method"
                    value={editing.form.payment_method}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: {
                          ...prev.form,
                          payment_method: event.target.value
                        }
                      }))
                    }
                  >
                    {PAYMENT_METHODS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm font-medium text-slate-700">
                  Internet service
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    name="internet_service"
                    value={editing.form.internet_service}
                    onChange={(event) =>
                      setEditing((prev) => ({
                        ...prev,
                        form: {
                          ...prev.form,
                          internet_service: event.target.value
                        }
                      }))
                    }
                  >
                    {INTERNET_SERVICES.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loading ? "Atualizando..." : "Atualizar predição"}
                </button>
                <button
                  type="button"
                  onClick={() => setEditing(null)}
                  className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50"
                >
                  Cancelar
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </div>
  );
}
