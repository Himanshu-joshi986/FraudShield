import React, { useEffect, useState, useCallback } from "react";
import { api } from "./services/api";
import CaseFileList from "./components/CaseFileList";
import CommandCenter from "./pages/CommandCenter";
import CaseDetail from "./pages/CaseDetail";
import Analytics from "./pages/Analytics";
import KnowledgeBase from "./pages/KnowledgeBase";
import NewCase from "./pages/NewCase";
import DataIntake from "./pages/DataIntake";

const NAV = [
  { id: "home", label: "Command Center" },
  { id: "analytics", label: "Analytics" },
  { id: "knowledge", label: "Knowledge Base" },
  { id: "data", label: "Data intake" },
];

export default function App() {
  const [page, setPage] = useState("home");
  const [activeCaseId, setActiveCaseId] = useState(null);
  const [cases, setCases] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showNewCase, setShowNewCase] = useState(false);
  const [appError, setAppError] = useState("");

  const refresh = useCallback(async () => {
    const [caseResult, analyticsResult] = await Promise.allSettled([
      api.listCases({ limit: 100 }),
      api.analytics(),
    ]);
    const errors = [];
    if (caseResult.status === "fulfilled") setCases(caseResult.value);
    else errors.push(`Cases: ${caseResult.reason.message}`);
    if (analyticsResult.status === "fulfilled") setAnalytics(analyticsResult.value);
    else errors.push(`Analytics: ${analyticsResult.reason.message}`);
    setAppError(errors.join(" · "));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  async function handleSeed() {
    setLoading(true);
    setAppError("");
    try {
      await api.seedSynthetic();
      await refresh();
    } catch (error) {
      setAppError(`Could not load the project case file: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleCaseCreated(caseId) {
    setShowNewCase(false);
    await refresh();
    openCase(caseId);
  }

  function openCase(caseId) {
    setActiveCaseId(caseId);
    setPage("case");
  }

  function goHome() {
    setActiveCaseId(null);
    setPage("home");
    refresh();
  }

  function navigateTo(nextPage) {
    setActiveCaseId(null);
    setPage(nextPage);
    refresh();
  }

  return (
    <div className="workspace-shell flex min-h-screen bg-ink-950">
      <CaseFileList cases={cases} activeCaseId={activeCaseId} onSelect={openCase} onGoHome={goHome} />

      <div className="workspace-main flex-1 flex flex-col">
        <div className="workspace-nav flex items-center gap-1 px-6 pt-4 border-b border-ink-750">
          <div className="flex gap-1">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => navigateTo(n.id)}
                className={`text-sm font-mono px-3 py-2 -mb-px border-b-2 transition-colors ${
                  page === n.id ? "border-forensic text-forensic" : "border-transparent text-ink-dim hover:text-ink-text"
                }`}
              >
                {n.label}
              </button>
            ))}
          </div>
          <button
            onClick={() => setShowNewCase(true)}
            className="ml-auto mb-2 text-xs font-mono px-3 py-2 border border-forensic text-forensic hover:bg-forensic/10 transition-colors"
          >
            + File new case
          </button>
        </div>

        {appError && <div className="app-error" role="alert"><strong>Connection problem</strong><span>{appError}</span><button onClick={refresh}>Retry</button></div>}

        {page === "home" && (
          <CommandCenter cases={cases} analytics={analytics} onSeed={handleSeed} onOpenCase={openCase} loading={loading} />
        )}
        {page === "case" && activeCaseId && <CaseDetail caseId={activeCaseId} onBack={goHome} onCaseUpdated={refresh} />}
        {page === "analytics" && <Analytics analytics={analytics} error={appError} />}
        {page === "knowledge" && <KnowledgeBase />}
        {page === "data" && <DataIntake onImported={refresh} />}
      </div>
      {showNewCase && (
        <NewCase onCancel={() => setShowNewCase(false)} onCreated={handleCaseCreated} />
      )}
    </div>
  );
}
