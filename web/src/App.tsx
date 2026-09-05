import { useEffect, useMemo, useState } from "react";
import {
  explainMatch,
  fetchMatch,
  fetchMetrics,
  fetchRecent,
  fetchUpcoming,
  MatchCard,
  pct,
  predictedPct,
} from "./api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";

type Mode = "fans" | "analyst";

function ProbBars({ p }: { p: MatchCard["probabilities"] }) {
  const rows = [
    { key: "home", label: "Home", value: p.home, cls: "home" },
    { key: "draw", label: "Draw", value: p.draw, cls: "draw" },
    { key: "away", label: "Away", value: p.away, cls: "away" },
  ];
  return (
    <div className="prob-row">
      {rows.map((r) => (
        <div className="prob" key={r.key}>
          <span>{r.label}</span>
          <div className={`bar ${r.cls}`}>
            <i style={{ width: `${Math.max(2, r.value * 100)}%` }} />
          </div>
          <strong>{pct(r.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function Badge({ m }: { m: MatchCard }) {
  if (m.status === "scheduled" || m.outcome == null) {
    return <span className="badge soon">Upcoming</span>;
  }
  if (m.correct === true) return <span className="badge ok">Correct</span>;
  if (m.correct === false) return <span className="badge miss">Missed</span>;
  return null;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("fans");
  const [upcoming, setUpcoming] = useState<MatchCard[]>([]);
  const [recent, setRecent] = useState<MatchCard[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<MatchCard | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [oddsNote, setOddsNote] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const [u, r] = await Promise.all([fetchUpcoming(40), fetchRecent(40)]);
        if (cancelled) return;
        setUpcoming(u.matches);
        setRecent(r.matches);
        setOddsNote((u as any).odds_note || null);
        // Prefer Everton vs Man United weekend if present, else first upcoming
        const everton = u.matches.find(
          (m) =>
            m.home_team === "Everton" &&
            String(m.away_team).includes("United") &&
            m.date.startsWith("2026-09-06")
        );
        const first = everton?.match_id || u.matches[0]?.match_id || r.matches[0]?.match_id || null;
        setSelectedId(first);
        setError(null);
      } catch (e: any) {
        if (!cancelled) setError(e.message || String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    (async () => {
      try {
        const d = await fetchMatch(selectedId);
        if (!cancelled) setDetail(d);
      } catch (e: any) {
        if (!cancelled) setError(e.message || String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (mode !== "analyst") return;
    fetchMetrics()
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, [mode]);

  const list = mode === "fans" ? upcoming : recent.length ? recent : upcoming;

  const shapChart = useMemo(() => {
    const top = detail?.shap?.top_features || [];
    return top.map((t) => ({
      name: t.label || t.feature,
      shap: t.shap,
    }));
  }, [detail]);

  async function onExplain() {
    if (!selectedId) return;
    setExplaining(true);
    try {
      const res = await explainMatch(selectedId);
      setDetail((d) => (d ? { ...d, explanation: res.explanation } : d));
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setExplaining(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1 className="brand">
            Pitch<span>Pulse</span>
          </h1>
          <p className="tagline">
            Premier League predictions with real footy signal — xG, shots, possession —
            plus plain-English explanations.
          </p>
        </div>
        <div className="mode-toggle" role="tablist">
          <button
            className={mode === "fans" ? "active" : ""}
            onClick={() => setMode("fans")}
          >
            Fans
          </button>
          <button
            className={mode === "analyst" ? "active" : ""}
            onClick={() => setMode("analyst")}
          >
            Analyst
          </button>
        </div>
      </header>

      {loading && <p className="loading">Loading fixtures…</p>}
      {error && <p className="error">{error}</p>}

      {!loading && (
        <div className={`grid ${mode === "fans" ? "two" : ""}`}>
          <section className="card">
            <h2>{mode === "fans" ? "This week's slate" : "Recent results"}</h2>
            {mode === "fans" && oddsNote && <p className="odds-note">{oddsNote}</p>}
            <div className="match-list">
              {list.map((m) => (
                <button
                  key={m.match_id}
                  className={`match-item ${selectedId === m.match_id ? "selected" : ""}`}
                  onClick={() => setSelectedId(m.match_id)}
                >
                  <div className="meta">
                    <span>
                      {m.date}
                      {m.kickoff ? ` · ${m.kickoff}` : ""}
                    </span>
                    <Badge m={m} />
                  </div>
                  <div className="teams">
                    {m.home_team} vs {m.away_team}
                  </div>
                  <div className="meta" style={{ marginTop: "0.35rem" }}>
                    <span>
                      Form {m.home_form || "—"} / {m.away_form || "—"}
                    </span>
                    <span className="pick">
                      {m.prediction_label || m.prediction} · {predictedPct(m)}
                    </span>
                  </div>
                </button>
              ))}
              {!list.length && (
                <p className="muted">
                  No fixtures yet. Run <code>python run_pipeline.py</code> then restart the API.
                </p>
              )}
            </div>
          </section>

          <section className="card">
            {!detail && <p className="muted">Select a match</p>}
            {detail && (
              <>
                <h2>
                  {detail.home_team} vs {detail.away_team}
                </h2>
                <div className="meta muted" style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                  <span>{detail.date}{detail.kickoff ? ` · ${detail.kickoff}` : ""}</span>
                  <Badge m={detail} />
                  {detail.outcome_label && <span>Actual: {detail.outcome_label}</span>}
                </div>

                <ProbBars p={detail.probabilities} />

                <div className="stat-chips">
                  <span className="chip">
                    Home xG (L5) <strong>{Number(detail.stats_snapshot?.home_xg_for_5 || 0).toFixed(2)}</strong>
                  </span>
                  <span className="chip">
                    Away xG (L5) <strong>{Number(detail.stats_snapshot?.away_xg_for_5 || 0).toFixed(2)}</strong>
                  </span>
                  <span className="chip">
                    Shots gap{" "}
                    <strong>
                      {(
                        Number(detail.stats_snapshot?.home_shots_5 || 0) -
                        Number(detail.stats_snapshot?.away_shots_5 || 0)
                      ).toFixed(1)}
                    </strong>
                  </span>
                  <span className="chip">
                    Poss %{" "}
                    <strong>
                      {Number(detail.stats_snapshot?.home_poss_5 || 50).toFixed(0)}–
                      {Number(detail.stats_snapshot?.away_poss_5 || 50).toFixed(0)}
                    </strong>
                  </span>
                </div>

                <div className="book-grid">
                  <div className="mini">
                    <h4>Our model</h4>
                    <div>H {pct(detail.probabilities.home)}</div>
                    <div>D {pct(detail.probabilities.draw)}</div>
                    <div>A {pct(detail.probabilities.away)}</div>
                  </div>
                  <div className="mini">
                    <h4>Bookmaker</h4>
                    {detail.bookmaker?.p_home != null ? (
                      <>
                        <div>H {pct(detail.bookmaker.p_home)}</div>
                        <div>D {pct(detail.bookmaker.p_draw)}</div>
                        <div>A {pct(detail.bookmaker.p_away)}</div>
                      </>
                    ) : (
                      <p className="muted" style={{ margin: 0 }}>
                        Pre-match odds aren’t in the historical CSV for unplayed games.
                        Add a free <code>THE_ODDS_API_KEY</code> to <code>.env</code> and
                        restart the API to pull live prices.
                      </p>
                    )}
                  </div>
                </div>

                {detail.explanation ? (
                  <p className="explain">{detail.explanation}</p>
                ) : (
                  <button className="primary" onClick={onExplain} disabled={explaining}>
                    {explaining ? "Writing explanation…" : "Explain this pick"}
                  </button>
                )}

                {mode === "analyst" && shapChart.length > 0 && (
                  <div style={{ marginTop: "1.25rem", height: 260 }}>
                    <h3>SHAP drivers</h3>
                    <ResponsiveContainer width="100%" height="85%">
                      <BarChart data={shapChart} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis type="number" stroke="#9bb5a8" />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={140}
                          stroke="#9bb5a8"
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip />
                        <Bar dataKey="shap" radius={4}>
                          {shapChart.map((e, i) => (
                            <Cell key={i} fill={e.shap >= 0 ? "#3ecf8e" : "#e85d5d"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}
          </section>

          {mode === "analyst" && (
            <section className="card" style={{ gridColumn: "1 / -1" }}>
              <h2>Model scoreboard</h2>
              {!metrics && <p className="muted">Metrics not loaded</p>}
              {metrics?.metrics && (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Log loss</th>
                      <th>Brier</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.metrics).map(([name, m]: any) =>
                      m && typeof m === "object" && "log_loss" in m ? (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{m.log_loss.toFixed(4)}</td>
                          <td>{m.brier.toFixed(4)}</td>
                          <td>{(m.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                      ) : null
                    )}
                  </tbody>
                </table>
              )}
              {metrics?.ablation?.delta && (
                <p className="muted" style={{ marginTop: "1rem" }}>
                  Ablation (full vs base): log-loss delta{" "}
                  <strong style={{ color: "var(--ink)" }}>
                    {metrics.ablation.delta.log_loss_full_minus_base?.toFixed?.(4) ??
                      metrics.ablation.delta.log_loss_full_minus_base}
                  </strong>{" "}
                  (negative = xG/shots/possession helped)
                </p>
              )}
              {metrics?.backtest && (
                <p className="muted">
                  Flat ROI {((metrics.backtest.flat_roi || 0) * 100).toFixed(1)}% · Kelly ROI{" "}
                  {((metrics.backtest.kelly_roi || 0) * 100).toFixed(1)}% ·{" "}
                  {metrics.backtest.n_bets} bets
                </p>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
