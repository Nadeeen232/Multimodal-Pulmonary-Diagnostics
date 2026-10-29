"""Visual layer for the Streamlit UI: color tokens, one CSS block and small HTML helpers.

Colors mirror .streamlit/config.toml ([theme.light] / [theme.dark]) so custom blocks match
Streamlit's own widgets. Cards are keyed containers (st.container(key="pr-...")), which
Streamlit exposes as stable `st-key-<key>` classes. Every helper escapes the text it receives.
"""

import html

TOKENS = {
    "light": {
        "bg": "#F4F3EF",
        "card": "#FFFFFF",
        "sunken": "#EEEDE8",
        "text": "#222322",
        "muted": "#686B68",
        "border": "#E3E3DE",
        "border-strong": "#CFCFC8",
        "accent": "#D6F36A",
        "accent-hover": "#CAE95A",
        "on-accent": "#222322",
        "focus": "#222323",
        "ink": "#222323",
        "ink-border": "#222323",
        "on-ink": "#F4F2ED",
        "ink-muted": "#A6AAA5",
        "lav-soft": "#F1EDFB",
        "lav-border": "#DED5F4",
        "normal": "#1E7A46",
        "normal-soft": "rgba(30, 122, 70, 0.10)",
        "abnormal": "#A3520A",
        "abnormal-soft": "rgba(163, 82, 10, 0.10)",
        "normal-zone": "rgba(30, 122, 70, 0.24)",
        "abnormal-zone": "rgba(163, 82, 10, 0.24)",
        "danger": "#B42318",
        "danger-soft": "rgba(180, 35, 24, 0.07)",
        "frame": "#161718",
        "shadow": "0 1px 2px rgba(34, 35, 34, 0.04), 0 10px 30px -12px rgba(34, 35, 34, 0.10)",
        "logo-ring": "transparent",
    },
    "dark": {
        "bg": "#171819",
        "card": "#242526",
        "sunken": "#2B2D2C",
        "text": "#F4F2ED",
        "muted": "#A6AAA5",
        "border": "#393B39",
        "border-strong": "#4A4D4A",
        "accent": "#D6F36A",
        "accent-hover": "#E2F98A",
        "on-accent": "#1B1C1A",
        "focus": "#D6F36A",
        "ink": "#101111",
        "ink-border": "#393B39",
        "on-ink": "#F4F2ED",
        "ink-muted": "#A6AAA5",
        "lav-soft": "#2A2833",
        "lav-border": "#423D55",
        "normal": "#86D9A2",
        "normal-soft": "rgba(134, 217, 162, 0.13)",
        "abnormal": "#F2B45A",
        "abnormal-soft": "rgba(242, 180, 90, 0.14)",
        "normal-zone": "rgba(134, 217, 162, 0.30)",
        "abnormal-zone": "rgba(242, 180, 90, 0.30)",
        "danger": "#F97066",
        "danger-soft": "rgba(249, 112, 102, 0.10)",
        "frame": "#0E0F0F",
        "shadow": "0 1px 2px rgba(0, 0, 0, 0.35)",
        "logo-ring": "#393B39",
    },
}

# The verdict sits on the graphite panel in both themes, so its colors are fixed.
INK_STATUS = {"normal": "#86D9A2", "abnormal": "#F2B45A"}

MONO = '"Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

CSS = """
/*VARS*/
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeaderActionElements"] { display: none; }
/* Markdown blocks carry -1rem to absorb a trailing <p> margin; the pr- blocks have none. */
[data-testid="stMarkdownContainer"]:has(> [class^="pr-"]) { margin-bottom: 0; }
/* The style and script blocks render nothing; drop the gap they leave. */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style),
[data-testid="stElementContainer"]:has(script) { display: none; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1380px; padding: 3.25rem 2rem 3rem;
}

/* Header */
.pr-top {
  display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem 2rem;
  flex-wrap: wrap; padding-bottom: 0.5rem;
}
.pr-brand { display: flex; align-items: center; gap: 1.1rem; }
.pr-logo {
  flex: none; width: 3.25rem; height: 3.25rem; border-radius: 17%; display: block;
  box-shadow: 0 0 0 1px var(--pr-logo-ring);
}
.pr-eyebrow {
  font-family: MONO; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--pr-muted);
}
.pr-top h1 {
  margin: 0.2rem 0 0.15rem; padding: 0; font-size: clamp(2rem, 3.2vw, 2.7rem); font-weight: 600;
  letter-spacing: -0.035em; line-height: 1.05; color: var(--pr-text);
}
.pr-top p { margin: 0; font-size: 0.95rem; color: var(--pr-muted); }
.pr-badge {
  display: inline-flex; align-items: center; gap: 0.55rem; padding: 0.45rem 0.85rem;
  border: 1px solid var(--pr-border); border-radius: 999px; background: var(--pr-card);
  font-size: 0.82rem; font-weight: 500; color: var(--pr-text);
}
.pr-badge .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--pr-muted); }
.pr-badge.is-ready .dot { background: var(--pr-normal); }
.pr-badge.is-down .dot, .pr-badge.is-loading .dot { background: var(--pr-abnormal); }
.pr-badge.is-down, .pr-badge.is-loading { color: var(--pr-muted); }
.pr-badge code {
  font-family: MONO; font-size: 0.75rem; color: var(--pr-muted); background: none; padding: 0;
}

/* Cards */
.st-key-pr-study, .st-key-pr-viewer, .st-key-pr-report, .st-key-pr-influence {
  background: var(--pr-card); border: 1px solid var(--pr-border); border-radius: 20px;
  padding: 1.35rem 1.4rem 1.45rem; box-shadow: var(--pr-shadow);
}
.st-key-pr-influence { background: var(--pr-lav-soft); border-color: var(--pr-lav-border); }
.pr-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.pr-head h3 {
  margin: 0; padding: 0; font-size: 1.08rem; font-weight: 600; letter-spacing: -0.015em;
  line-height: 1.3; color: var(--pr-text);
}
.pr-head p { margin: 0.15rem 0 0; font-size: 0.82rem; color: var(--pr-muted); }
.pr-label { font-size: 0.875rem; font-weight: 500; color: var(--pr-text); margin-bottom: -0.35rem; }

/* Inputs and buttons */
[data-testid="stFileUploaderDropzone"] {
  background: var(--pr-sunken); border: 1.5px dashed var(--pr-border-strong); border-radius: 14px;
}
[data-testid="stBaseButton-primary"]:not(:disabled) {
  background: var(--pr-accent); border-color: var(--pr-accent); color: var(--pr-on-accent);
}
[data-testid="stBaseButton-primary"]:not(:disabled):hover {
  background: var(--pr-accent-hover); border-color: var(--pr-accent-hover); color: var(--pr-on-accent);
}
[data-testid="stBaseButton-primary"]:not(:disabled) p { color: var(--pr-on-accent); }
.st-key-pr-analyze button { min-height: 3.1rem; border-radius: 12px; }
.st-key-pr-analyze button p { font-weight: 600; font-size: 0.95rem; }
.st-key-sample-normal button { background: var(--pr-normal-soft); color: var(--pr-normal); border-color: transparent; }
.st-key-sample-abnormal button { background: var(--pr-abnormal-soft); color: var(--pr-abnormal); border-color: transparent; }
.st-key-sample-false_prediction button { background: var(--pr-sunken); color: var(--pr-text); border-color: transparent; }
[class*="st-key-sample-"] button p { color: inherit; font-weight: 500; }
[class*="st-key-sample-"] button:hover { filter: brightness(0.97); }
/*SELECTED*/
button:focus-visible, .pr-copy:focus-visible { outline: 2px solid var(--pr-focus); outline-offset: 2px; }

/* Radiographs */
.st-key-pr-preview [data-testid="stImage"] img, .st-key-pr-viewer [data-testid="stImage"] img {
  background: var(--pr-frame); border-radius: 14px; object-fit: contain; width: 100%;
}
.st-key-pr-preview [data-testid="stImage"] img { max-height: 420px; }
.st-key-pr-viewer [data-testid="stImage"] img { max-height: 560px; }
[data-testid="stImageCaption"] { color: var(--pr-muted); font-size: 0.8rem; }
.pr-frame {
  height: 13rem; border-radius: 14px; background: var(--pr-frame); display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 0.35rem;
  color: #A6AAA5; font-size: 0.85rem; text-align: center; padding: 1rem;
}
.pr-frame span { font-family: MONO; font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase; color: #7D817C; }
.pr-caption { font-size: 0.8rem; color: var(--pr-muted); margin-top: -0.35rem; }

/* Assessment */
.pr-assess { display: grid; grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr); gap: 1rem; }
.pr-score {
  background: var(--pr-ink); color: var(--pr-on-ink); border: 1px solid var(--pr-ink-border);
  border-radius: 20px; padding: 1.35rem 1.45rem 1.4rem; display: flex; flex-direction: column;
  justify-content: space-between; gap: 1.25rem; min-height: 15rem;
}
.pr-score .pr-eyebrow { color: var(--pr-ink-muted); }
.pr-score-top { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
.pr-pill {
  display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.32rem 0.8rem;
  border-radius: 999px; font-weight: 600; font-size: 0.9rem;
}
.pr-pill::before { content: ""; width: 0.45rem; height: 0.45rem; border-radius: 50%; background: currentColor; }
.pr-big {
  font-family: MONO; font-weight: 500; font-size: clamp(3.1rem, 5.2vw, 4.4rem);
  letter-spacing: -0.06em; line-height: 0.9; color: var(--pr-on-ink);
}
.pr-big .pt { margin: 0 -0.2em; }
.pr-big small { font-size: 0.5em; letter-spacing: -0.02em; margin-left: 0.08em; color: var(--pr-ink-muted); }
.pr-score-label { margin-top: 0.55rem; font-size: 0.85rem; color: var(--pr-ink-muted); }
.pr-score-rel {
  margin-top: 0.9rem; padding-top: 0.75rem; border-top: 1px solid rgba(244, 242, 237, 0.12);
  font-family: MONO; font-size: 0.74rem; color: var(--pr-on-ink);
}
.pr-thr {
  background: var(--pr-card); border: 1px solid var(--pr-border); border-radius: 20px;
  padding: 1.35rem 1.45rem 1.25rem; display: flex; flex-direction: column; gap: 0.9rem;
  box-shadow: var(--pr-shadow);
}
.pr-thr-top { display: flex; align-items: baseline; justify-content: space-between; gap: 0.75rem; }
.pr-thr-top b { font-family: MONO; font-weight: 500; font-size: 1.5rem; letter-spacing: -0.04em; color: var(--pr-text); }
.pr-scale { position: relative; height: 2.9rem; }
.pr-scale .bar {
  position: absolute; left: 0; right: 0; top: 0.95rem; height: 0.55rem; border-radius: 999px;
  overflow: hidden; display: flex;
}
.pr-scale .lo { background: var(--pr-normal-zone); }
.pr-scale .hi { flex: 1; background: var(--pr-abnormal-zone); }
.pr-scale .tick {
  position: absolute; top: 0.45rem; width: 2px; height: 1.55rem; border-radius: 2px;
  background: var(--pr-text); transform: translateX(-1px);
}
.pr-scale .marker {
  position: absolute; top: 0.68rem; width: 1.1rem; height: 1.1rem; border-radius: 50%;
  transform: translateX(-50%); border: 3px solid var(--pr-card);
}
.pr-scale .marker.is-normal { background: var(--pr-normal); box-shadow: 0 0 0 1px var(--pr-normal); }
.pr-scale .marker.is-abnormal { background: var(--pr-abnormal); box-shadow: 0 0 0 1px var(--pr-abnormal); }
.pr-scale .ends {
  position: absolute; left: 0; right: 0; bottom: 0; display: flex; justify-content: space-between;
  font-family: MONO; font-size: 0.7rem; color: var(--pr-muted);
}
.pr-key { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; font-size: 0.78rem; color: var(--pr-muted); }
.pr-key i {
  display: inline-block; width: 0.65rem; height: 0.65rem; border-radius: 3px; margin-right: 0.35rem;
  vertical-align: -1px;
}
.pr-reason { font-size: 0.86rem; line-height: 1.55; color: var(--pr-muted); border-top: 1px solid var(--pr-border); padding-top: 0.8rem; }
.pr-reason strong { color: var(--pr-text); font-weight: 600; }
.pr-reason code { font-family: MONO; font-size: 0.75rem; background: none; padding: 0; color: var(--pr-muted); }
.pr-note {
  display: flex; align-items: center; gap: 0.65rem; padding: 0.7rem 1rem; border-radius: 14px;
  background: var(--pr-sunken); font-size: 0.84rem; color: var(--pr-text);
}
.pr-note b {
  flex: none; width: 1.25rem; height: 1.25rem; border-radius: 50%; display: grid; place-items: center;
  font-family: MONO; font-size: 0.7rem; background: var(--pr-text); color: var(--pr-bg);
}

/* Text influence */
.pr-tokens { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.pr-token {
  display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.32rem 0.55rem 0.32rem 0.7rem;
  border-radius: 10px; background: var(--pr-card); border: 1px solid var(--pr-lav-border);
  font-size: 0.87rem; color: var(--pr-text);
}
.pr-token b { font-family: MONO; font-weight: 500; font-size: 0.76rem; }
.pr-token.up b { color: var(--pr-abnormal); }
.pr-token.down b { color: var(--pr-normal); }
.pr-token em { font-style: normal; font-size: 0.68rem; color: var(--pr-muted); }
.pr-legend { font-size: 0.76rem; line-height: 1.5; color: var(--pr-muted); }

/* Report */
.pr-meta { font-family: MONO; font-size: 0.74rem; color: var(--pr-muted); margin-top: 0.25rem; }
.pr-copy {
  font: inherit; font-size: 0.82rem; font-weight: 500; padding: 0.38rem 0.85rem; border-radius: 10px;
  border: 1px solid var(--pr-border); background: var(--pr-card); color: var(--pr-text); cursor: pointer;
}
.pr-copy:hover { background: var(--pr-sunken); }
.pr-doc section { padding: 0.95rem 0; border-top: 1px solid var(--pr-border); }
.pr-doc section:first-child { border-top: 0; padding-top: 0.2rem; }
.pr-doc section:last-child { padding-bottom: 0; }
.pr-doc h4 {
  margin: 0 0 0.35rem; padding: 0; font-family: MONO; font-size: 0.68rem; font-weight: 500;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--pr-muted);
}
.pr-doc p { margin: 0; font-size: 0.95rem; line-height: 1.68; color: var(--pr-text); max-width: 70ch; }

/* States */
.pr-empty {
  background: var(--pr-card); border: 1px solid var(--pr-border); border-radius: 20px;
  padding: 1.6rem 1.6rem 1.7rem; box-shadow: var(--pr-shadow);
}
.pr-empty h3 { margin: 0.35rem 0 0.3rem; padding: 0; font-size: 1.35rem; font-weight: 600; letter-spacing: -0.025em; color: var(--pr-text); }
.pr-empty > p { margin: 0; color: var(--pr-muted); font-size: 0.92rem; max-width: 52ch; }
.pr-empty.is-warn h3 { color: var(--pr-abnormal); }
.pr-steps { list-style: none; display: grid; gap: 0.55rem; margin: 1.2rem 0 0; padding: 0; }
.pr-steps li { display: flex; align-items: center; gap: 0.75rem; font-size: 0.9rem; color: var(--pr-text); margin: 0; }
.pr-steps b {
  flex: none; width: 1.8rem; height: 1.8rem; border-radius: 9px; display: grid; place-items: center;
  font-family: MONO; font-size: 0.72rem; font-weight: 500; background: var(--pr-sunken); color: var(--pr-text);
}
.pr-ghost { display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 0.75rem; margin-top: 1.5rem; }
.pr-ghost div {
  border: 1.5px dashed var(--pr-border-strong); border-radius: 14px; height: 6.5rem; padding: 0.75rem 0.85rem;
  font-family: MONO; font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--pr-muted);
}
.pr-ghost .wide { grid-column: 1 / -1; height: 11rem; }
.pr-skel { display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 1rem; }
.pr-skel div { height: 15rem; border-radius: 20px; background: var(--pr-sunken); animation: pr-pulse 1.4s ease-in-out infinite; }
.pr-skel .wide { grid-column: 1 / -1; height: 22rem; }
.pr-skel .ink { background: var(--pr-ink); opacity: 0.85; }
@keyframes pr-pulse { 50% { opacity: 0.5; } }
.pr-error {
  border: 1px solid var(--pr-danger); border-radius: 16px; padding: 1rem 1.15rem;
  background: var(--pr-danger-soft); color: var(--pr-text); font-size: 0.9rem; line-height: 1.55;
}
.pr-error strong { display: block; color: var(--pr-danger); margin-bottom: 0.2rem; }

.pr-foot {
  margin-top: 2.25rem; padding-top: 1.1rem; border-top: 1px solid var(--pr-border);
  font-size: 0.82rem; color: var(--pr-muted);
}
.pr-reveal { animation: pr-fade 180ms ease-out; }
@keyframes pr-fade { from { opacity: 0; transform: translateY(4px); } }

@media (prefers-reduced-motion: reduce) {
  .pr-skel div, .pr-reveal { animation: none; }
}
@media (max-width: 1100px) {
  .st-key-pr-workspace > * > [data-testid="stHorizontalBlock"],
  .st-key-pr-details > * > [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  .st-key-pr-workspace > * > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  .st-key-pr-details > * > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important; width: 100% !important; min-width: 100%;
  }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 3.5rem 1rem 2rem; }
  .pr-assess, .pr-ghost, .pr-skel { grid-template-columns: minmax(0, 1fr); }
  .st-key-pr-study, .st-key-pr-viewer, .st-key-pr-report, .st-key-pr-influence { padding: 1.1rem 1rem 1.2rem; border-radius: 18px; }
  .pr-score { min-height: 0; }
  .pr-logo { width: 2.75rem; height: 2.75rem; }
}
""".replace("MONO", MONO)


def _variables(name):
    return " ".join(f"--pr-{key}: {value};" for key, value in TOKENS[name].items())


def css(theme_type, selected_sample=None):
    """Both palettes; PAGE_SCRIPT switches them live when the viewer changes theme.

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
    selected = ""
    if selected_sample:
        selected = (
            f".st-key-sample-{selected_sample} button "
            "{ border-color: currentColor; box-shadow: inset 0 0 0 1px currentColor; }"
        )
    return (
        "<style>"
        + CSS.replace("/*VARS*/", variables).replace("/*SELECTED*/", selected)
        + "</style>"
    )


# Tags <html> with the theme Streamlit actually painted, and wires the report Copy button.
PAGE_SCRIPT = r"""
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
  if (window.__prWired) return;
  window.__prWired = true;
  const watch = new MutationObserver(() => requestAnimationFrame(apply));
  watch.observe(document.head, { childList: true });
  watch.observe(document.body, { attributes: true, attributeFilter: ["class", "style"], subtree: true });
  document.addEventListener("click", async (event) => {
    const button = event.target.closest(".pr-copy");
    if (!button) return;
    const text = button.dataset.copy || "";
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const area = Object.assign(document.createElement("textarea"), { value: text });
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });
})();
</script>
"""


def esc(value):
    return html.escape(str(value))


def header(state, version, logo_uri=""):
    labels = {"ready": "Model ready", "down": "Service unavailable", "loading": "Model not ready"}
    version_html = f"<code>{esc(version)}</code>" if version else ""
    return (
        '<div class="pr-top"><div class="pr-brand">'
        f'<img class="pr-logo" src="{esc(logo_uri)}" alt=""><div>'
        '<div class="pr-eyebrow">Pulmonary Review Assistant</div>'
        "<h1>Pulmonary Review</h1><p>Multimodal chest X-ray assessment</p></div></div>"
        f'<div class="pr-badge is-{state}" role="status"><span class="dot"></span>'
        f"{labels[state]} {version_html}</div></div>"
    )


def card_head(title, subtitle="", extra=""):
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return f'<div class="pr-head"><div><h3>{esc(title)}</h3>{sub}</div>{extra}</div>'


def label(text):
    return f'<div class="pr-label">{esc(text)}</div>'


def caption(text):
    return f'<div class="pr-caption">{esc(text)}</div>'


def frame_placeholder(title="No radiograph", body="Upload an X-ray or load a sample case"):
    return f'<div class="pr-frame"><span>{esc(title)}</span>{esc(body)}</div>'


def empty(title, body, warn=False):
    steps = ""
    if not warn:
        steps = (
            '<ol class="pr-steps">'
            "<li><b>01</b>Upload a chest X-ray, or load a sample case</li>"
            "<li><b>02</b>Enter the clinical indication</li>"
            "<li><b>03</b>Select Analyze Study</li></ol>"
            '<div class="pr-ghost" aria-hidden="true"><div>Verdict and score</div>'
            '<div>Decision threshold</div><div class="wide">Radiograph and heatmap</div>'
            "<div>Text influence</div><div>Generated report</div></div>"
        )
    return (
        f'<div class="pr-empty{" is-warn" if warn else ""}">'
        '<div class="pr-eyebrow">Assessment</div>'
        f"<h3>{esc(title)}</h3><p>{esc(body)}</p>{steps}</div>"
    )


def error(message):
    return (
        f'<div class="pr-error" role="alert"><strong>Analysis failed</strong>{esc(message)}</div>'
    )


def skeleton():
    return (
        '<div class="pr-skel" aria-label="Analyzing"><div class="ink"></div><div></div>'
        '<div class="wide"></div></div>'
    )


def percent(value, digits=1):
    return f"{value * 100:.{digits}f}".rstrip("0").rstrip(".") + "%"


def assessment(prediction, predicted_class):
    score = min(max(float(prediction["probability"]), 0.0), 1.0)
    threshold = min(max(float(prediction["threshold"]), 0.0), 1.0)
    kind = "abnormal" if predicted_class == "Abnormal" else "normal"
    tier = prediction["priority_tier"].replace("_", " ").capitalize()
    color = INK_STATUS[kind]
    whole, tenth = f"{score * 100:.1f}".split(".")
    relation = "At or above" if score >= threshold else "Below"
    return (
        '<div class="pr-assess pr-reveal">'
        '<div class="pr-score"><div class="pr-score-top">'
        '<span class="pr-eyebrow">Model assessment</span>'
        f'<span class="pr-pill" style="color:{color};background:{color}24">{esc(predicted_class)}</span>'
        f'</div><div><div class="pr-big">{whole}<span class="pt">.</span>{tenth}<small>%</small></div>'
        '<div class="pr-score-label">Abnormality score, uncalibrated</div>'
        f'<div class="pr-score-rel">{relation} the {percent(threshold)} threshold</div></div></div>'
        '<div class="pr-thr"><div class="pr-thr-top"><span class="pr-eyebrow">Decision threshold</span>'
        f"<b>{percent(threshold)}</b></div>"
        '<div class="pr-scale" role="img" '
        f'aria-label="Score {score * 100:.1f}% against threshold {percent(threshold)}">'
        f'<div class="bar"><div class="lo" style="width:{threshold * 100:.2f}%"></div><div class="hi"></div></div>'
        f'<div class="tick" style="left:{threshold * 100:.2f}%"></div>'
        f'<div class="marker is-{kind}" style="left:{score * 100:.2f}%"></div>'
        '<div class="ends"><span>0%</span><span>100%</span></div></div>'
        '<div class="pr-key"><span><i style="background:var(--pr-normal-zone)"></i>'
        f"Normal below {percent(threshold)}</span>"
        '<span><i style="background:var(--pr-abnormal-zone)"></i>'
        f"Abnormal at {percent(threshold)} or above</span></div>"
        f'<div class="pr-reason"><strong>{esc(tier)}.</strong> {esc(prediction["priority_reason"])}<br>'
        "Uncalibrated score against weak annotation labels, not the probability of disease. "
        f"Model <code>{esc(prediction['model_version'])}</code>.</div></div></div>"
    )


def note(text):
    return f'<div class="pr-note"><b aria-hidden="true">i</b><span>{esc(text)}</span></div>'


def tokens(highlights, limit=8):
    """Largest score changes first; values are the API's contributions in percentage points."""
    ranked = sorted(highlights, key=lambda item: abs(item["contribution"]), reverse=True)[:limit]
    chips = []
    for item in ranked:
        kind = "up" if item["contribution"] > 0 else "down"
        field = " <em>comparison</em>" if item["field"] == "comparison" else ""
        chips.append(
            f'<span class="pr-token {kind}">{esc(item["token"])}{field}'
            f"<b>{item['contribution'] * 100:+.2f}</b></span>"
        )
    return f'<div class="pr-tokens">{"".join(chips)}</div>'


LEGEND = (
    '<div class="pr-legend">Change in the abnormality score, in percentage points, when the word '
    "is removed. Positive values raised the score.</div>"
)


def report_head(text):
    first = str(text).strip().split("\n", 1)[0]
    meta = (
        f'<div class="pr-meta">{esc(first)}</div>' if first.startswith("Predicted Class:") else ""
    )
    button = f'<button type="button" class="pr-copy" data-copy="{esc(text)}">Copy</button>'
    return f'<div class="pr-head"><div><h3>Generated report</h3>{meta}</div>{button}</div>'


def report(text):
    """Render the rule-based report's 'Heading:' blocks; the predicted class line is in the head."""
    sections = []
    for block in str(text).strip().split("\n\n"):
        heading, _, body = block.partition("\n")
        if not body and ":" in heading:
            continue  # "Predicted Class: ..." one-liner
        sections.append(
            f"<section><h4>{esc(heading.rstrip(':'))}</h4><p>{esc(body.strip())}</p></section>"
        )
    if not sections:
        sections.append(f"<section><p>{esc(text)}</p></section>")
    return f'<div class="pr-doc pr-reveal">{"".join(sections)}</div>'


def footer():
    return (
        '<div class="pr-foot">Research prototype trained on weak labels from the Indiana '
        "University chest X-ray collection. Not clinically validated.</div>"
    )
