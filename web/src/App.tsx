import { useEffect, useMemo, useState } from "react";
import {
  explainMatch,
  fetchMatch,
  fetchMetrics,
  fetchRecent,
  fetchUpcoming,
  formatEdge,
  HitRate,
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

const TEAM_SEARCH_ALIASES: Record<string, string> = {
  united: "man united",
  "man u": "man united",
  mufc: "man united",
  city: "man city",
  "man city": "man city",
  spurs: "tottenham",
  arsenal: "arsenal",
  chelsea: "chelsea",
  liverpool: "liverpool",
  wolves: "wolves",
  forest: "nott'm forest",
  nottingham: "nott'm forest",
  villa: "aston villa",
  palace: "crystal palace",
  hammers: "west ham",
  toon: "newcastle",
  magpies: "newcastle",
};

function teamNeedle(query: string): string {
  const q = query.trim().toLowerCase();
  if (!q) return "";
  return TEAM_SEARCH_ALIASES[q] ?? q;
}

function matchIncludesTeam(m: MatchCard, needle: string): boolean {
  if (!needle) return true;
  const home = (m.home_team || "").toLowerCase();
  const away = (m.away_team || "").toLowerCase();
  return home.includes(needle) || away.includes(needle);
}

function byNewestFirst(a: MatchCard, b: MatchCard): number {
  const da = a.date || "";
  const db = b.date || "";
  if (da !== db) return db.localeCompare(da);
  return (b.kickoff || "15:00").localeCompare(a.kickoff || "15:00");
}

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

function EdgePill({ m }: { m: MatchCard }) {
  const label = formatEdge(m.edge);
  if (!label) return null;
  const pos = (m.edge?.pp ?? 0) > 0;
  const neg = (m.edge?.pp ?? 0) < 0;
  return (
    <span className={`badge edge ${pos ? "pos" : neg ? "neg" : ""}`} title="Model vs bookmaker on predicted side">
      Edge {label}
    </span>
  );
}

function buildShareText(m: MatchCard): string {
  const edge = formatEdge(m.edge);
  const lines = [
    `PitchPulse pick`,
    `${m.home_team} vs ${m.away_team}`,
    `${m.date}${m.kickoff ? ` · ${m.kickoff}` : ""}`,
    `${m.prediction_label || m.prediction} · ${predictedPct(m)}`,
    `Model H/D/A: ${pct(m.probabilities.home)} / ${pct(m.probabilities.draw)} / ${pct(m.probabilities.away)}`,
  ];
  if (m.bookmaker?.p_home != null) {
    lines.push(
      `Book H/D/A: ${pct(m.bookmaker.p_home)} / ${pct(m.bookmaker.p_draw)} / ${pct(m.bookmaker.p_away)}`
    );
  }
  if (edge) lines.push(`Edge vs market: ${edge}`);
  if (m.scorelines?.length) {
    lines.push(`Top scorelines: ${m.scorelines.slice(0, 3).map((s) => `${s.score} (${pct(s.probability)})`).join(", ")}`);
  }
  if (m.explanation) lines.push("", m.explanation);
  lines.push("", "#PitchPulse #PremierLeague");
  return lines.join("\n");
}

export default function App() {
  const [mode, setMode] = useState<Mode>("fans");
  const [upcoming, setUpcoming] = useState<MatchCard[]>([]);
  const [recent, setRecent] = useState<MatchCard[]>([]);
  const [recentSeason, setRecentSeason] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<MatchCard | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [oddsNote, setOddsNote] = useState<string | null>(null);
  const [hitRate, setHitRate] = useState<HitRate | null>(null);
  const [loading, setLoading] = useState(true);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [teamQuery, setTeamQuery] = useState("");
  const [matchweek, setMatchweek] = useState<number | "all">("all");
  const [shareMsg, setShareMsg] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const [u, r] = await Promise.all([fetchUpcoming(80), fetchRecent(40)]);
        if (cancelled) return;
        setUpcoming(u.matches);
        setRecent(r.matches);
        setHitRate(r.hit_rate || null);
        setRecentSeason((r as any).season || null);
        setOddsNote((u as any).odds_note || null);
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
        if (!cancelled) {
          setDetail(d);
          setError(null);
        }
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

  const needle = teamNeedle(teamQuery);
  const searching = Boolean(needle);

  const matchweeks = useMemo(() => {
    // Fans: upcoming GWs only. Analyst: completed GWs only. Avoid mixing future weeks into results.
    const source = searching
      ? [...upcoming, ...recent]
      : mode === "fans"
        ? upcoming
        : recent;
    const set = new Set<number>();
    for (const m of source) {
      if (m.matchday != null && !Number.isNaN(Number(m.matchday))) set.add(Number(m.matchday));
    }
    const weeks = Array.from(set).sort((a, b) => a - b);
    // Analyst: newest completed GW first in the chip row
    return mode === "analyst" && !searching ? weeks.slice().reverse() : weeks;
  }, [mode, upcoming, recent, searching]);

  // Analyst shows the full current-season slate (newest first), not a single GW.
  useEffect(() => {
    if (searching) return;
    setMatchweek("all");
  }, [mode, searching]);

  // Switching to Analyst should land on the newest result, not a future Fans pick.
  useEffect(() => {
    if (mode !== "analyst" || searching || !recent.length) return;
    const inRecent = recent.some((m) => m.match_id === selectedId);
    if (!inRecent) setSelectedId(recent[0].match_id);
  }, [mode, recent, searching, selectedId]);

  const clubHub = useMemo(() => {
    if (!searching) return null;
    const clubMatches = [...upcoming, ...recent].filter((m) => matchIncludesTeam(m, needle));
    const played = clubMatches
      .filter((m) => m.outcome != null)
      .sort((a, b) => (a.date || "").localeCompare(b.date || ""));
    const form = played
      .slice(-5)
      .map((m) => {
        const isHome = (m.home_team || "").toLowerCase().includes(needle);
        if (m.outcome === "D") return "D";
        if (isHome) return m.outcome === "H" ? "W" : "L";
        return m.outcome === "A" ? "W" : "L";
      })
      .join("-");
    const next5 = clubMatches
      .filter((m) => m.status === "scheduled" || m.outcome == null)
      .sort((a, b) => (a.date || "").localeCompare(b.date || ""))
      .slice(0, 5);
    const label = teamQuery.trim() || needle;
    return { label, form: form || "—", next5, nPlayed: played.length };
  }, [searching, needle, teamQuery, upcoming, recent]);

  const list = useMemo(() => {
    let base: MatchCard[];
    if (searching) {
      const byId = new Map<string, MatchCard>();
      for (const m of [...upcoming, ...recent]) {
        if (matchIncludesTeam(m, needle)) byId.set(m.match_id, m);
      }
      base = Array.from(byId.values()).sort(byNewestFirst);
    } else if (mode === "fans") {
      base = upcoming;
    } else {
      base = [...(recent.length ? recent : upcoming)].sort(byNewestFirst);
    }
    if (matchweek !== "all") {
      base = base.filter((m) => Number(m.matchday) === matchweek);
    }
    return base;
  }, [mode, upcoming, recent, needle, searching, matchweek]);

  const listTitle = searching
    ? `${teamQuery.trim()} fixtures`
    : mode === "fans"
      ? "This week's slate"
      : recentSeason
        ? `Recent results · ${recentSeason}`
        : "Recent results";

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

  async function onShare() {
    if (!detail) return;
    const text = buildShareText(detail);
    try {
      await navigator.clipboard.writeText(text);
      setShareMsg("Copied pick to clipboard");
    } catch {
      setShareMsg("Could not copy — select and copy manually");
    }
    setTimeout(() => setShareMsg(null), 2500);
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
          <button className={mode === "fans" ? "active" : ""} onClick={() => setMode("fans")}>
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
        <div className={`grid ${mode === "fans" || searching ? "two" : ""}`}>
          <section className="card">
            <h2>{listTitle}</h2>

            <div className="team-search">
              <input
                type="search"
                placeholder="Search club — e.g. United, Spurs, Villa"
                value={teamQuery}
                onChange={(e) => setTeamQuery(e.target.value)}
                aria-label="Search fixtures by club"
              />
              {teamQuery && (
                <button type="button" className="clear-search" onClick={() => setTeamQuery("")}>
                  Clear
                </button>
              )}
            </div>

            {matchweeks.length > 0 && (
              <div className="mw-chips" role="group" aria-label="Matchweek filter">
                <button
                  type="button"
                  className={matchweek === "all" ? "active" : ""}
                  onClick={() => setMatchweek("all")}
                >
                  All GW
                </button>
                {matchweeks.map((mw) => (
                  <button
                    key={mw}
                    type="button"
                    className={matchweek === mw ? "active" : ""}
                    onClick={() => setMatchweek(mw)}
                  >
                    GW{mw}
                  </button>
                ))}
              </div>
            )}

            {mode === "fans" && !searching && oddsNote && <p className="odds-note">{oddsNote}</p>}
            {mode === "analyst" && !searching && (
              <p className="odds-note">
                Graded picks for the current season, newest first. Older matchweeks stay in
                the list — use the GW chips if you want to filter.
              </p>
            )}
            {searching && (
              <p className="odds-note">Club view: upcoming and recent for this team.</p>
            )}

            {clubHub && (
              <div className="club-hub">
                <div>
                  <strong>{clubHub.label}</strong>
                  <span className="muted"> · form {clubHub.form}</span>
                </div>
                {clubHub.next5.length > 0 && (
                  <div className="club-next">
                    {clubHub.next5.map((m) => (
                      <button
                        key={m.match_id}
                        type="button"
                        className="chip linkish"
                        onClick={() => setSelectedId(m.match_id)}
                      >
                        {m.date.slice(5)} {m.home_team} vs {m.away_team}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {mode === "analyst" && !searching && hitRate && hitRate.strip?.length > 0 && (
              <div className="hit-rate">
                <span>
                  Last {hitRate.last_n}:{" "}
                  <strong>
                    {hitRate.last_n_correct}/{hitRate.last_n}
                  </strong>{" "}
                  ({pct(hitRate.last_n_rate)})
                </span>
                <div className="hit-strip">
                  {hitRate.strip.map((s) => (
                    <i
                      key={s.match_id}
                      className={s.correct ? "ok" : "miss"}
                      title={s.correct ? "Correct" : "Missed"}
                    />
                  ))}
                </div>
              </div>
            )}

            <div className="match-list">
              {list.map((m) => (
                <button
                  key={m.match_id}
                  className={`match-item ${selectedId === m.match_id ? "selected" : ""}`}
                  onClick={() => setSelectedId(m.match_id)}
                >
                  <div className="meta">
                    <span>
                      {m.matchday != null ? `GW${m.matchday} · ` : ""}
                      {m.date}
                      {m.kickoff ? ` · ${m.kickoff}` : ""}
                    </span>
                    <span className="badge-row">
                      <EdgePill m={m} />
                      <Badge m={m} />
                    </span>
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
                  {searching ? (
                    "No fixtures for that club in the loaded slate."
                  ) : (
                    <>
                      No fixtures yet. Run <code>python run_pipeline.py</code> then restart the API.
                    </>
                  )}
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
                <div
                  className="meta muted"
                  style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}
                >
                  <span>
                    {detail.matchday != null ? `GW${detail.matchday} · ` : ""}
                    {detail.date}
                    {detail.kickoff ? ` · ${detail.kickoff}` : ""}
                  </span>
                  <Badge m={detail} />
                  <EdgePill m={detail} />
                  {detail.outcome_label && <span>Actual: {detail.outcome_label}</span>}
                </div>

                <ProbBars p={detail.probabilities} />

                <div className="stat-chips">
                  <span className="chip">
                    Home xG (L5){" "}
                    <strong>{Number(detail.stats_snapshot?.home_xg_for_5 || 0).toFixed(2)}</strong>
                  </span>
                  <span className="chip">
                    Away xG (L5){" "}
                    <strong>{Number(detail.stats_snapshot?.away_xg_for_5 || 0).toFixed(2)}</strong>
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
                        Live book odds attach when the Odds API cache has this fixture.
                      </p>
                    )}
                  </div>
                </div>

                {detail.scorelines && detail.scorelines.length > 0 && (
                  <div className="scorelines">
                    <h3>Likely scorelines</h3>
                    <p className="odds-note">Independent Poisson from L5 xG (not the classifier).</p>
                    <div className="score-row">
                      {detail.scorelines.map((s) => (
                        <span key={s.score} className="chip">
                          {s.score} <strong>{pct(s.probability)}</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="action-row">
                  {detail.explanation ? (
                    <p className="explain">{detail.explanation}</p>
                  ) : (
                    <button className="primary" onClick={onExplain} disabled={explaining}>
                      {explaining ? "Writing explanation…" : "Explain this pick"}
                    </button>
                  )}
                  <button type="button" className="secondary" onClick={onShare}>
                    Share pick
                  </button>
                  {shareMsg && <span className="share-msg">{shareMsg}</span>}
                </div>

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

          {mode === "analyst" && !searching && (
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
                      <th>Calibration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.metrics)
                      .filter(([name]) => !name.startsWith("_"))
                      .map(([name, m]: any) =>
                      m && typeof m === "object" && "log_loss" in m ? (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{m.log_loss.toFixed(4)}</td>
                          <td>{m.brier.toFixed(4)}</td>
                          <td>{(m.accuracy * 100).toFixed(1)}%</td>
                          <td>{m.calibration || "—"}</td>
                        </tr>
                      ) : null
                    )}
                  </tbody>
                </table>
              )}
              {metrics?.metrics?._meta?.best_model && (
                <p className="muted" style={{ marginTop: "0.75rem" }}>
                  Best production model:{" "}
                  <strong style={{ color: "var(--ink)" }}>
                    {metrics.metrics._meta.best_model}
                  </strong>
                  {" · "}hold-out from {metrics.metrics._meta.test_season_start}+
                </p>
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
