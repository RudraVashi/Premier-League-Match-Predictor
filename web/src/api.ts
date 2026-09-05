export type Probabilities = { home: number; draw: number; away: number };

export type Edge = {
  pp: number;
  model_p: number;
  book_p: number;
  side?: string | null;
} | null;

export type Scoreline = {
  home_goals: number;
  away_goals: number;
  score: string;
  probability: number;
};

export type HitRate = {
  n: number;
  correct: number;
  rate: number | null;
  last_n: number;
  last_n_correct: number;
  last_n_rate: number | null;
  strip: Array<{ match_id: string; correct: boolean | null; label: string }>;
};

export type MatchCard = {
  match_id: string;
  date: string;
  kickoff?: string | null;
  season?: string;
  matchday?: number | null;
  home_team: string;
  away_team: string;
  status?: string;
  home_form?: string | null;
  away_form?: string | null;
  prediction?: string | null;
  prediction_label?: string | null;
  outcome?: string | null;
  outcome_label?: string | null;
  correct?: boolean | null;
  probabilities: Probabilities;
  bookmaker?: {
    odds_home?: number | null;
    odds_draw?: number | null;
    odds_away?: number | null;
    p_home?: number | null;
    p_draw?: number | null;
    p_away?: number | null;
  } | null;
  edge?: Edge;
  scorelines?: Scoreline[];
  stats_snapshot?: Record<string, number | null | undefined>;
  shap?: {
    predicted_class?: string;
    probabilities?: Record<string, number>;
    top_features?: Array<{
      feature: string;
      label?: string;
      value: number;
      shap: number;
    }>;
    model?: string;
  };
  explanation?: string | null;
};

const API_BASE = import.meta.env.VITE_API_URL || "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export function fetchUpcoming(limit = 40) {
  return get<{ count: number; matches: MatchCard[]; odds_note?: string }>(
    `/fixtures/upcoming?limit=${limit}&days=45`
  );
}

export function fetchRecent(limit = 40) {
  return get<{ count: number; matches: MatchCard[]; hit_rate?: HitRate; season?: string }>(
    `/fixtures/recent?limit=${limit}`
  );
}

export function fetchMatch(id: string) {
  return get<MatchCard>(`/match/${encodeURIComponent(id)}`);
}

export async function explainMatch(id: string) {
  const res = await fetch(`${API_BASE}/match/${encodeURIComponent(id)}/explain`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ explanation: string; cached: boolean }>;
}

export function fetchMetrics() {
  return get<any>("/metrics");
}

export function pct(n?: number | null) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

/** Probability of the model's predicted class (H/D/A). */
export function predictedPct(m: {
  prediction?: string | null;
  probabilities?: Probabilities | null;
}) {
  const p = m.probabilities;
  if (!p) return "—";
  const key = (m.prediction || "").toUpperCase();
  if (key === "H") return pct(p.home);
  if (key === "D") return pct(p.draw);
  if (key === "A") return pct(p.away);
  return pct(Math.max(p.home, p.draw, p.away));
}

export function formatEdge(edge?: Edge) {
  if (!edge || edge.pp == null) return null;
  const sign = edge.pp > 0 ? "+" : "";
  return `${sign}${edge.pp.toFixed(1)}pp`;
}
