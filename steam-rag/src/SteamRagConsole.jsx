/**
 * SteamRagConsole React component
 * --------------------------------
 * Drop into any React app (Vite/Next). Tailwind classes are used but standard utility CSS works as well.
 * Update API_BASE_URL if your FastAPI backend uses a different origin.
 */
import React, { useCallback, useEffect, useMemo, useState } from "react";

const API_BASE_URL = "http://localhost:8000";
const STORAGE_KEY = "steam-rag-console-state";
const SENTIMENT_OPTIONS = ["very_negative", "negative", "neutral", "positive", "very_positive"];
const DEFAULT_STATE = {
  common: { appId: "1364780", dateStart: "", dateEnd: "", topics: "", sentiments: [], rerank: false },
  search: { query: "", k: 20 },
  ask: { question: "", k: 64 },
  counts: { topic: "", groupBy: "", minHelpful: "" },
};
const INPUT_CLASS = "rounded border border-slate-300 p-2";
const toArrayFromCSV = (value) => (value ? value.split(",").map((part) => part.trim()).filter(Boolean) : []);
const truthy = (value) => value !== undefined && value !== null && value !== "";
const formatDateTime = (value) => {
  if (!value) return "";
  try {
    return new Date(value).toLocaleString();
  } catch (err) {
    return value;
  }
};
async function fetchJSON(path, { method = "GET", body } = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch (err) {
      const error = new Error("Invalid JSON response");
      error.status = response.status;
      error.data = raw;
      throw error;
    }
  }
  if (!response.ok) {
    const error = new Error(`Request failed with status ${response.status}`);
    error.status = response.status;
    error.data = data ?? raw;
    throw error;
  }
  return data;
}
function usePersistentState() {
  const [state, setState] = useState(DEFAULT_STATE);
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setState({
          common: { ...DEFAULT_STATE.common, ...(parsed.common || {}) },
          search: { ...DEFAULT_STATE.search, ...(parsed.search || {}) },
          ask: { ...DEFAULT_STATE.ask, ...(parsed.ask || {}) },
          counts: { ...DEFAULT_STATE.counts, ...(parsed.counts || {}) },
        });
      }
    } catch (err) {
      console.warn("Failed to restore state", err);
    }
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);
  return [state, setState];
}
const ErrorBanner = ({ error }) => {
  if (!error) return null;
  const details = error.data && typeof error.data === "object" ? JSON.stringify(error.data, null, 2) : error.data || error.message;
  return (
    <div className="mt-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
      <p className="font-semibold">{error.message}</p>
      {error.status && <p>Status: {error.status}</p>}
      {details && <pre className="mt-2 whitespace-pre-wrap break-all text-xs">{details}</pre>}
      {error instanceof TypeError && <p className="mt-2">Check API availability and CORS.</p>}
    </div>
  );
};
const Field = ({ label, children, className = "" }) => (
  <label className={`flex flex-col gap-1 text-sm ${className}`}>
    <span>{label}</span>
    {children}
  </label>
);
const SentimentMultiSelect = ({ value, onChange }) => (
  <select multiple value={value} onChange={(event) => onChange(Array.from(event.target.selectedOptions, (option) => option.value))} className={INPUT_CLASS}>
    {SENTIMENT_OPTIONS.map((option) => (
      <option key={option} value={option}>
        {option}
      </option>
    ))}
  </select>
);
const MetaList = ({ item }) => {
  const rows = [
    { label: "Review ID", value: item.review_id },
    { label: "Created", value: formatDateTime(item.created_at) },
    { label: "Score", value: truthy(item.score) ? item.score?.toFixed(4) : null },
    { label: "Helpful", value: item.helpful_count ?? 0 },
    { label: "Funny", value: item.funny_count ?? 0 },
  ].filter((row) => truthy(row.value) || row.label === "Helpful" || row.label === "Funny");
  return (
    <dl className="grid gap-1 text-sm">
      {rows.map((row) => (
        <div key={row.label} className="flex gap-2">
          <dt className="font-semibold">{row.label}:</dt>
          <dd className="truncate">{row.value}</dd>
        </div>
      ))}
      {item.review_url && (
        <div className="flex gap-2">
          <dt className="font-semibold">Link:</dt>
          <dd>
            <a href={item.review_url} target="_blank" rel="noreferrer" className="text-blue-600 underline">
              Open Review
            </a>
          </dd>
        </div>
      )}
    </dl>
  );
};
const SearchResults = ({ results }) => {
  if (!results) return null;
  if (!results.length) return <p className="mt-4 text-sm text-slate-600">No results.</p>;
  return (
    <ul className="mt-4 space-y-4">
      {results.map((item) => (
        <li key={item.review_id} className="rounded border border-slate-200 p-4">
          <MetaList item={item} />
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer text-slate-700">View Text</summary>
            <p className="mt-2 whitespace-pre-wrap text-slate-800">{item.review_text}</p>
          </details>
        </li>
      ))}
    </ul>
  );
};
const SnippetList = ({ snippets }) => {
  if (!snippets?.length) return null;
  return (
    <ul className="mt-4 space-y-4">
      {snippets.map((item) => (
        <li key={item.review_id} className="rounded border border-indigo-200 bg-indigo-50 p-4">
          <MetaList item={item} />
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer text-slate-700">View Snippet</summary>
            <p className="mt-2 whitespace-pre-wrap text-slate-800">{item.review_text}</p>
          </details>
        </li>
      ))}
    </ul>
  );
};
const tableClass = "border border-emerald-200 px-2 py-1";
const MiniTable = ({ columns, rows }) => (
  <div className="mt-2 overflow-x-auto">
    <table className="min-w-full border-collapse text-xs">
      <thead>
        <tr className="bg-emerald-100">
          {columns.map((col) => (
            <th key={col.key} className={`${tableClass} ${col.align === "right" ? "text-right" : "text-left"}`}>
              {col.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={row.key ?? index}>
            {columns.map((col) => (
              <td key={col.key} className={`${tableClass} ${col.align === "right" ? "text-right" : "text-left"}`}>
                {col.render ? col.render(row) : row[col.key] ?? 0}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);
const CountsTable = ({ data }) => {
  if (!data) return null;
  if (!data.length) return <p className="mt-4 text-sm text-slate-600">No counts available.</p>;
  return (
    <div className="mt-4 space-y-6">
      {data.map((entry, index) => {
        const sentiments = Object.keys(entry.by_sentiment || {});
        const sentimentColumns = [
          { key: "sentiment", label: "Sentiment" },
          { key: "count", label: "Count", align: "right" },
        ];
        const sentimentRows = sentiments.map((sentiment) => ({ key: sentiment, sentiment, count: entry.by_sentiment?.[sentiment] ?? 0 }));
        const bucketColumns = [
          { key: "bucket", label: "Bucket" },
          { key: "total", label: "Total", align: "right" },
          ...sentiments.map((sentiment) => ({ key: sentiment, label: sentiment, align: "right", render: (row) => row.by_sentiment?.[sentiment] ?? 0 })),
        ];
        const bucketRows = (entry.buckets || []).map((bucket) => ({ ...bucket, key: bucket.bucket }));
        return (
          <div key={index} className="rounded border border-emerald-200 bg-emerald-50 p-4 text-sm">
            <h4 className="font-semibold">Topic: {entry.topic ?? "(all)"}</h4>
            <p className="mt-1">Total: {entry.total}</p>
            <MiniTable columns={sentimentColumns} rows={sentimentRows} />
            {bucketRows.length > 0 && <MiniTable columns={bucketColumns} rows={bucketRows} />}
          </div>
        );
      })}
    </div>
  );
};
const InputControl = ({ config, value, onChange }) => {
  if (config.render) return config.render(value, onChange);
  if (config.component === "textarea")
    return <textarea rows={config.rows || 2} className={INPUT_CLASS} value={value} onChange={(event) => onChange(event.target.value)} placeholder={config.placeholder} />;
  if (config.component === "select")
    return (
      <select className={INPUT_CLASS} value={value} onChange={(event) => onChange(event.target.value)}>
        {config.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  return <input type={config.type || "text"} min={config.min} list={config.list} className={INPUT_CLASS} value={value} onChange={(event) => onChange(event.target.value)} placeholder={config.placeholder} />;
};

export default function SteamRagConsole() {
  const [state, setState] = usePersistentState();
  const { common, search, ask, counts } = state;
  const topicsArray = useMemo(() => toArrayFromCSV(common.topics), [common.topics]);
  const sentimentsArray = useMemo(() => common.sentiments || [], [common.sentiments]);
  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [askResult, setAskResult] = useState(null),
    [askLoading, setAskLoading] = useState(false),
    [askError, setAskError] = useState(null),
    [countsResult, setCountsResult] = useState(null),
    [countsLoading, setCountsLoading] = useState(false),
    [countsError, setCountsError] = useState(null);
  const updateState = useCallback((section, patch) => setState((prev) => ({ ...prev, [section]: { ...prev[section], ...patch } })), [setState]);
  const runSearch = useCallback(async () => {
    setSearchLoading(true);
    setSearchError(null);
    try {
      const body = { query: search.query, app_id: common.appId, lang: "english", k: Number(search.k) || 20, rerank: !!common.rerank };
      if (common.dateStart) body.date_start = common.dateStart;
      if (common.dateEnd) body.date_end = common.dateEnd;
      if (topicsArray.length) body.topics = topicsArray;
      if (sentimentsArray.length) body.sentiments = sentimentsArray;
      const data = await fetchJSON("/search", { method: "POST", body });
      setSearchResults(data?.results || []);
    } catch (err) {
      console.error("Search failed", err);
      setSearchError(err);
      setSearchResults(null);
    } finally {
      setSearchLoading(false);
    }
  }, [common, search, topicsArray, sentimentsArray]);
  const runAsk = useCallback(async () => {
    setAskLoading(true);
    setAskError(null);
    try {
      const body = { query: ask.question, app_id: common.appId, k: Number(ask.k) || 64, rerank: !!common.rerank };
      if (common.dateStart) body.date_start = common.dateStart;
      if (common.dateEnd) body.date_end = common.dateEnd;
      if (topicsArray.length) body.topics = topicsArray;
      if (sentimentsArray.length) body.sentiments = sentimentsArray;
      const data = await fetchJSON("/ask", { method: "POST", body });
      setAskResult(data);
    } catch (err) {
      console.error("Ask failed", err);
      setAskError(err);
      setAskResult(null);
    } finally {
      setAskLoading(false);
    }
  }, [ask, common, topicsArray, sentimentsArray]);
  const runCounts = useCallback(async () => {
    setCountsLoading(true);
    setCountsError(null);
    try {
      const params = new URLSearchParams();
      if (common.appId) params.set("app_id", common.appId);
      if (counts.topic) params.set("topic", counts.topic);
      if (counts.groupBy) params.set("group_by", counts.groupBy);
      if (truthy(counts.minHelpful)) params.set("min_helpful", counts.minHelpful);
      if (common.dateStart) params.set("date_start", common.dateStart);
      if (common.dateEnd) params.set("date_end", common.dateEnd);
      const data = await fetchJSON(`/counts${params.toString() ? `?${params}` : ""}`);
      setCountsResult(data);
    } catch (err) {
      console.error("Counts failed", err);
      setCountsError(err);
      setCountsResult(null);
    } finally {
      setCountsLoading(false);
    }
  }, [common, counts]);

  const commonFields = [
    {
      key: "appId",
      label: "App ID",
      render: (value, onChange) => (
        <>
          <input type="text" list="app-id-options" className={INPUT_CLASS} value={value} onChange={(event) => onChange(event.target.value)} />
          <datalist id="app-id-options">
            <option value="1364780">Street Fighter 6 (1364780)</option>
          </datalist>
        </>
      ),
    },
    { key: "dateStart", label: "Date Start (YYYY-MM-DD)", placeholder: "2025-01-01" },
    { key: "dateEnd", label: "Date End (YYYY-MM-DD)", placeholder: "2025-09-18" },
    { key: "topics", label: "Topics (comma-separated)", placeholder: "netcode, combat" },
  ];
  const searchFields = [{ key: "query", label: "Query", component: "textarea", rows: 2, placeholder: "connection rollback netcode issues", colSpan: 2 }, { key: "k", label: "Top K", type: "number", min: 1 }];
  const askFields = [{ key: "question", label: "Question", component: "textarea", rows: 3, placeholder: "Summarize stability complaints since launch", colSpan: 2 }, { key: "k", label: "Top K", type: "number", min: 1 }];
  const countsFields = [{ key: "topic", label: "Topic", placeholder: "netcode" }, { key: "groupBy", label: "Group By", component: "select", options: [{ value: "", label: "(none)" }, { value: "month", label: "Month" }] }, { key: "minHelpful", label: "Min Helpful", type: "number", min: 0, placeholder: "0" }];

  return (
    <div className="mx-auto max-w-5xl space-y-8 p-4 text-slate-800">
      <section className="rounded border border-slate-200 p-4 shadow-sm">
        <h2 className="text-lg font-semibold">Common Controls</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {commonFields.map((config) => (
            <Field key={config.key} label={config.label} className={config.colSpan === 2 ? "md:col-span-2" : ""}>
              <InputControl config={config} value={common[config.key]} onChange={(value) => updateState("common", { [config.key]: value })} />
            </Field>
          ))}
          <Field label="Sentiments" className="md:col-span-2">
            <SentimentMultiSelect value={common.sentiments} onChange={(sentiments) => updateState("common", { sentiments })} />
          </Field>
          <label className="flex items-center gap-2 text-sm md:col-span-2">
            <input type="checkbox" checked={common.rerank} onChange={(event) => updateState("common", { rerank: event.target.checked })} />
            <span>Enable reranker</span>
          </label>
        </div>
      </section>

      <section className="rounded border border-slate-200 p-4 shadow-sm">
        <h2 className="text-lg font-semibold">Search</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {searchFields.map((config) => (
            <Field key={config.key} label={config.label} className={config.colSpan === 2 ? "md:col-span-2" : ""}>
              <InputControl config={config} value={search[config.key]} onChange={(value) => updateState("search", { [config.key]: value })} />
            </Field>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button className="rounded bg-slate-800 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" onClick={runSearch} disabled={searchLoading}>
            Run Search
          </button>
          {searchLoading && <span className="text-sm text-slate-600">Loading…</span>}
        </div>
        <ErrorBanner error={searchError} />
        <SearchResults results={searchResults} />
      </section>

      <section className="rounded border border-slate-200 p-4 shadow-sm">
        <h2 className="text-lg font-semibold">Ask</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {askFields.map((config) => (
            <Field key={config.key} label={config.label} className={config.colSpan === 2 ? "md:col-span-2" : ""}>
              <InputControl config={config} value={ask[config.key]} onChange={(value) => updateState("ask", { [config.key]: value })} />
            </Field>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" onClick={runAsk} disabled={askLoading}>
            Ask
          </button>
          {askLoading && <span className="text-sm text-slate-600">Loading…</span>}
        </div>
        <ErrorBanner error={askError} />
        {askResult && (
          <div className="mt-4 rounded border border-indigo-200 bg-indigo-50 p-4 text-sm">
            <h3 className="font-semibold">Answer</h3>
            <p className="mt-2 whitespace-pre-wrap">{askResult.answer}</p>
            <div className="mt-4">
              <h4 className="font-semibold">Citations</h4>
              <div className="mt-1 flex flex-wrap gap-2 text-xs">
                {askResult.citations?.map((citation) => (
                  <span key={citation} className="rounded-full bg-indigo-200 px-2 py-1 text-indigo-800">
                    {citation}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-semibold">Snippets</h4>
              <SnippetList snippets={askResult.snippets} />
            </div>
          </div>
        )}
      </section>

      <section className="rounded border border-slate-200 p-4 shadow-sm">
        <h2 className="text-lg font-semibold">Counts</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {countsFields.map((config) => (
            <Field key={config.key} label={config.label} className={config.colSpan === 2 ? "md:col-span-2" : ""}>
              <InputControl config={config} value={counts[config.key]} onChange={(value) => updateState("counts", { [config.key]: value })} />
            </Field>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button className="rounded bg-emerald-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" onClick={runCounts} disabled={countsLoading}>
            Get Counts
          </button>
          {countsLoading && <span className="text-sm text-slate-600">Loading…</span>}
        </div>
        <ErrorBanner error={countsError} />
        <CountsTable data={countsResult} />
      </section>
    </div>
  );
}
