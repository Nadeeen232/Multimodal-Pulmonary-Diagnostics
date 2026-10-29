"""Visual layer for the Streamlit UI: color tokens, one CSS block and small HTML helpers.

Colors mirror .streamlit/config.toml ([theme.light] / [theme.dark]) so custom blocks match
Streamlit's own widgets. Every helper escapes the text it receives.
"""

import html

TOKENS = {
    "light": {
        "bg": "#F6F7F9",
        "surface": "#FCFCFD",
        "sunken": "#EEF1F4",
        "text": "#111827",
        "muted": "#5B6472",
        "border": "#E3E7EC",
        "accent": "#0F766E",
        "on-accent": "#FFFFFF",
        "normal": "#15803D",
        "normal-soft": "rgba(21, 128, 61, 0.10)",
        "abnormal": "#B45309",
        "abnormal-soft": "rgba(180, 83, 9, 0.10)",
        "up-soft": "rgba(180, 83, 9, 0.14)",
        "down-soft": "rgba(15, 118, 110, 0.12)",
        "frame": "#0C1014",
        "shadow": "0 1px 2px rgba(17, 24, 39, 0.05), 0 8px 24px rgba(17, 24, 39, 0.05)",
    },
    "dark": {
        "bg": "#0C1014",
        "surface": "#11161C",
        "sunken": "#141A21",
        "text": "#E6EAF0",
        "muted": "#9AA4B2",
        "border": "#232B35",
        "accent": "#2DD4BF",
        "on-accent": "#04201D",
        "normal": "#4ADE80",
        "normal-soft": "rgba(74, 222, 128, 0.12)",
        "abnormal": "#F59E0B",
        "abnormal-soft": "rgba(245, 158, 11, 0.14)",
        "up-soft": "rgba(245, 158, 11, 0.18)",
        "down-soft": "rgba(45, 212, 191, 0.16)",
        "frame": "#05080B",
        "shadow": "0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px rgba(0, 0, 0, 0.25)",
    },
}

CSS = """
%(vars)s
[data-testid="stHeader"] { background: transparent; }
/* The style and theme-script blocks render nothing; drop the gap they leave. */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style),
[data-testid="stElementContainer"]:has(script) { display: none; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem;
}
h1, h2, h3 { letter-spacing: -0.01em; }

.pr-header {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  flex-wrap: wrap; padding-bottom: 1rem; margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--pr-border);
}
.pr-brand { display: flex; flex-direction: column; gap: 0.15rem; }
.pr-brand strong { font-size: 1.35rem; font-weight: 600; color: var(--pr-text); }
.pr-brand span { font-size: 0.875rem; color: var(--pr-muted); }
.pr-badge {
  display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.75rem;
  border: 1px solid var(--pr-border); border-radius: 999px; background: var(--pr-surface);
  font-size: 0.8125rem; color: var(--pr-text);
}
.pr-badge .pr-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%%; background: var(--pr-muted); }
.pr-badge.is-ready .pr-dot { background: var(--pr-normal); }
.pr-badge.is-down .pr-dot { background: var(--pr-abnormal); }
.pr-badge code, .pr-mono {
  font-family: "Geist Mono", ui-monospace, monospace; font-size: 0.78rem;
  color: var(--pr-muted); background: none; padding: 0;
}

.pr-title { font-size: 1rem; font-weight: 600; color: var(--pr-text); margin: 0.25rem 0 0.75rem; }
.pr-hint { font-size: 0.8125rem; color: var(--pr-muted); margin: 0.25rem 0 0.5rem; }

[data-testid="stImage"] img {
  border-radius: 10px; background: var(--pr-frame); border: 1px solid var(--pr-border);
  max-height: 460px; object-fit: contain;
}
[data-testid="stBaseButton-primary"]:not(:disabled),
[data-testid="stBaseButton-primary"]:not(:disabled) p { color: var(--pr-on-accent); font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { border-style: dashed; }

.pr-empty {
  border: 1px dashed var(--pr-border); border-radius: 10px; padding: 2.5rem 1.5rem;
  text-align: center; color: var(--pr-muted); background: var(--pr-surface);
}
.pr-empty strong { display: block; color: var(--pr-text); font-weight: 600; margin-bottom: 0.35rem; }
.pr-empty.is-warn strong { color: var(--pr-abnormal); }

.pr-verdict {
  display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 0.75rem 1rem;
  padding: 1.25rem 1.25rem 1rem; border: 1px solid var(--pr-border); border-radius: 10px;
  background: var(--pr-surface); box-shadow: var(--pr-shadow);
}
.pr-pill {
  display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
  font-weight: 600; font-size: 0.9375rem;
}
.pr-pill.is-normal { color: var(--pr-normal); background: var(--pr-normal-soft); }
.pr-pill.is-abnormal { color: var(--pr-abnormal); background: var(--pr-abnormal-soft); }
.pr-score { text-align: right; }
.pr-score b {
  display: block; font-family: "Geist Mono", ui-monospace, monospace; font-weight: 500;
  font-size: 2.25rem; line-height: 1.1; color: var(--pr-text);
}
.pr-score span { font-size: 0.75rem; color: var(--pr-muted); }
.pr-scale { grid-column: 1 / -1; position: relative; height: 2.25rem; margin-top: 0.25rem; }
.pr-scale .track {
  position: absolute; left: 0; right: 0; top: 0.6rem; height: 2px; background: var(--pr-border);
}
.pr-scale .tick {
  position: absolute; top: 0.2rem; width: 2px; height: 0.8rem; background: var(--pr-muted);
  transform: translateX(-1px);
}
.pr-scale .marker {
  position: absolute; top: 0.3rem; width: 0.65rem; height: 0.65rem; border-radius: 50%%;
  transform: translateX(-50%%); border: 2px solid var(--pr-surface);
}
.pr-scale .marker.is-normal { background: var(--pr-normal); }
.pr-scale .marker.is-abnormal { background: var(--pr-abnormal); }
.pr-scale .labels {
  position: absolute; left: 0; right: 0; top: 1.15rem; display: flex; justify-content: space-between;
  font-family: "Geist Mono", ui-monospace, monospace; font-size: 0.72rem; color: var(--pr-muted);
}
.pr-scale .thr {
  position: absolute; top: 1.15rem; transform: translateX(-50%%); white-space: nowrap;
  font-family: "Geist Mono", ui-monospace, monospace; font-size: 0.72rem; color: var(--pr-text);
}
.pr-reason {
  grid-column: 1 / -1; font-size: 0.875rem; color: var(--pr-muted);
  border-top: 1px solid var(--pr-border); padding-top: 0.75rem;
}
.pr-reason strong { color: var(--pr-text); font-weight: 600; }

.pr-chips { line-height: 2.1; font-size: 0.9375rem; color: var(--pr-text); }
.pr-chip { padding: 0.15rem 0.35rem; border-radius: 6px; }
.pr-chip.up { background: var(--pr-up-soft); }
.pr-chip.down { background: var(--pr-down-soft); }
.pr-legend { font-size: 0.78rem; color: var(--pr-muted); margin-top: 0.25rem; }
.pr-legend i {
  display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 3px;
  vertical-align: -1px; margin: 0 0.3rem 0 0.6rem;
}

.pr-report { display: grid; gap: 0.9rem; }
.pr-report h4 {
  margin: 0 0 0.2rem; font-size: 0.8125rem; font-weight: 600; color: var(--pr-muted);
}
.pr-report p { margin: 0; color: var(--pr-text); line-height: 1.6; max-width: 70ch; }

.pr-error {
  border: 1px solid var(--pr-abnormal); border-radius: 10px; padding: 0.9rem 1rem;
  background: var(--pr-abnormal-soft); color: var(--pr-text); font-size: 0.9rem;
}
.pr-error strong { color: var(--pr-abnormal); }

.pr-skel { display: grid; gap: 0.75rem; }
.pr-skel div {
  height: 1rem; border-radius: 6px; background: var(--pr-sunken);
  animation: pr-pulse 1.4s ease-in-out infinite;
}
.pr-skel .block { height: 9rem; }
@keyframes pr-pulse { 50%% { opacity: 0.45; } }

.pr-foot {
  margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--pr-border);
  font-size: 0.8125rem; color: var(--pr-muted);
}

.pr-reveal { animation: pr-fade 180ms ease-out; }
@keyframes pr-fade { from { opacity: 0; transform: translateY(4px); } }

@media (prefers-reduced-motion: reduce) {
  .pr-skel div, .pr-reveal { animation: none; }
}
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"], .block-container { padding-top: 3.5rem; }
  .pr-score b { font-size: 1.85rem; }
  .pr-verdict { padding: 1rem; }
}
"""


def _variables(name):
    return " ".join(f"--pr-{key}: {value};" for key, value in TOKENS[name].items())


def css(theme_type):
    """Both palettes; THEME_SCRIPT switches them live when the viewer changes theme.

    The server-side theme only picks the default for the first paint, because Streamlit does
    not rerun the script when the theme changes in the browser.
    """
    first, other = ("dark", "light") if theme_type == "dark" else ("light", "dark")
    variables = "\n".join(
        [
            f":root {{ {_variables(first)} }}",
            f':root[data-pr-theme="{other}"] {{ {_variables(other)} }}',
            f':root[data-pr-theme="{first}"] {{ {_variables(first)} }}',
        ]
    )
    return "<style>" + CSS % {"vars": variables} + "</style>"


# Reads the background Streamlit actually painted and tags <html> with data-pr-theme.
THEME_SCRIPT = r"""
<script>
(() => {
  const apply = () => {
    const app = document.querySelector('[data-testid="stApp"]') || document.body;
    const rgb = (getComputedStyle(app).backgroundColor.match(/\d+/g) || []).map(Number);
    if (rgb.length < 3) return;
    const dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 128;
    document.documentElement.dataset.prTheme = dark ? "dark" : "light";
  };
  apply();
  if (window.__prThemeWatch) return;
  window.__prThemeWatch = new MutationObserver(() => requestAnimationFrame(apply));
  window.__prThemeWatch.observe(document.head, { childList: true });
  window.__prThemeWatch.observe(document.body, {
    attributes: true, attributeFilter: ["class", "style"], subtree: true,
  });
})();
</script>
"""


def esc(value):
    return html.escape(str(value))


def header(state, version):
    labels = {"ready": "Model ready", "down": "Service unavailable", "loading": "Model not ready"}
    classes = {"ready": "is-ready", "down": "is-down", "loading": ""}
    version_html = f"<code>{esc(version)}</code>" if version else ""
    return (
        '<div class="pr-header"><div class="pr-brand">'
        "<strong>Pulmonary Review Assistant</strong>"
        "<span>Chest X-ray and clinical indication, reviewed together</span></div>"
        f'<div class="pr-badge {classes[state]}"><span class="pr-dot"></span>'
        f"{labels[state]} {version_html}</div></div>"
    )


def empty(title, body, warn=False):
    return f'<div class="pr-empty{" is-warn" if warn else ""}"><strong>{esc(title)}</strong>{esc(body)}</div>'


def error(message):
    return f'<div class="pr-error"><strong>Analysis failed.</strong> {esc(message)}</div>'


def skeleton():
    return '<div class="pr-skel"><div class="block"></div><div></div><div style="width:70%"></div><div style="width:85%"></div></div>'


def percent(value, digits=1):
    return f"{value * 100:.{digits}f}".rstrip("0").rstrip(".") + "%"


def verdict(prediction, predicted_class):
    score = min(max(float(prediction["probability"]), 0.0), 1.0)
    threshold = min(max(float(prediction["threshold"]), 0.0), 1.0)
    state = "is-abnormal" if predicted_class == "Abnormal" else "is-normal"
    tier = prediction["priority_tier"].replace("_", " ").capitalize()
    return (
        '<div class="pr-verdict pr-reveal">'
        f'<div><span class="pr-pill {state}">{esc(predicted_class)}</span></div>'
        f'<div class="pr-score"><b>{score * 100:.1f}%</b><span>abnormality score</span></div>'
        '<div class="pr-scale"><div class="track"></div>'
        f'<div class="tick" style="left:{threshold * 100:.2f}%"></div>'
        f'<div class="marker {state}" style="left:{score * 100:.2f}%"></div>'
        '<div class="labels"><span>0%</span><span>100%</span></div>'
        f'<div class="thr" style="left:{threshold * 100:.2f}%">threshold {percent(threshold)}</div></div>'
        f'<div class="pr-reason"><strong>{esc(tier)}.</strong> {esc(prediction["priority_reason"])}</div>'
        "</div>"
    )


def chips(raw, highlights):
    """Indication text with attributed tokens highlighted; positions come from the API."""
    parts, cursor = [], 0
    for item in sorted(highlights, key=lambda h: h["start"]):
        parts.append(esc(raw[cursor : item["start"]]))
        kind = "up" if item["contribution"] > 0 else "down"
        parts.append(f'<span class="pr-chip {kind}">{esc(raw[item["start"] : item["end"]])}</span>')
        cursor = item["end"]
    parts.append(esc(raw[cursor:]))
    return f'<div class="pr-chips">{"".join(parts)}</div>'


LEGEND = (
    '<div class="pr-legend"><i style="background:var(--pr-up-soft)"></i>raises the score'
    '<i style="background:var(--pr-down-soft)"></i>lowers it or has no effect</div>'
)


def report(text):
    """Render the rule-based report's 'Heading:' blocks; the predicted class line is shown above."""
    sections = []
    for block in str(text).strip().split("\n\n"):
        heading, _, body = block.partition("\n")
        if not body and ":" in heading:
            continue  # "Predicted Class: ..." one-liner
        sections.append(f"<div><h4>{esc(heading.rstrip(':'))}</h4><p>{esc(body.strip())}</p></div>")
    if not sections:
        sections.append(f"<div><p>{esc(text)}</p></div>")
    return f'<div class="pr-report pr-reveal">{"".join(sections)}</div>'


def footer():
    return (
        '<div class="pr-foot">Research prototype trained on weak labels from the Indiana '
        "University chest X-ray collection. Not clinically validated.</div>"
    )
