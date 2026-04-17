// Global UI helpers. Page-specific behavior lives in inline <script> blocks
// inside each template; keep this file small and side-effect-free.

(function () {
  "use strict";

  // Expose a tiny helper namespace.
  window.DGA = window.DGA || {};

  /** Escape HTML for safe insertion via innerHTML. */
  DGA.escape = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c];
    });
  };

  /** POST JSON to a Flask endpoint, return parsed JSON response. */
  DGA.postJSON = async function (url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`POST ${url} → ${res.status}: ${text}`);
    }
    return res.json();
  };

  /** POST a chunked stream to a Flask endpoint, calling onLine for each line. */
  DGA.streamPost = async function (url, body, onLine) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new Error(`POST ${url} → ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 1);
        onLine(line);
      }
    }
    if (buf) onLine(buf);
  };

  /** Frame <-> time helpers for <video> scrubbing. */
  DGA.frameToTime = function (frame, fps) {
    return fps > 0 ? frame / fps : 0;
  };
  DGA.timeToFrame = function (time, fps) {
    return Math.round(time * fps);
  };

  /** Tab switcher: [role="tablist"] > [role="tab"][data-tab="X"] + [data-panel="X"]. */
  DGA.wireTabs = function (root) {
    const tabs = root.querySelectorAll('[role="tab"]');
    tabs.forEach((t) => {
      t.addEventListener("click", () => {
        const target = t.dataset.tab;
        tabs.forEach((tt) => tt.classList.toggle("active", tt === t));
        root
          .querySelectorAll("[data-panel]")
          .forEach((p) =>
            p.classList.toggle("active", p.dataset.panel === target)
          );
      });
    });
  };
})();
