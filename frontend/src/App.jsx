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

  const refresh = useCallback(async () => {
    const [c, a] = await Promise.all([api.listCases({ limit: 100 }), api.analytics()]);
    setCases(c);
    setAnalytics(a);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  async function handleSeed() {
    setLoading(true);
    try {
      await api.seedSynthetic();
      await refresh();
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

        {page === "home" && (
          <CommandCenter cases={cases} analytics={analytics} onSeed={handleSeed} onOpenCase={openCase} loading={loading} />
        )}
        {page === "case" && activeCaseId && <CaseDetail caseId={activeCaseId} onBack={goHome} onCaseUpdated={refresh} />}
        {page === "analytics" && <Analytics analytics={analytics} />}
        {page === "knowledge" && <KnowledgeBase />}
        {page === "data" && <DataIntake onImported={refresh} />}
      </div>
      {showNewCase && (
        <NewCase onCancel={() => setShowNewCase(false)} onCreated={handleCaseCreated} />
      )}
    </div>
  );
}
