# JANK Template

- **Title:** Karn dashboard HTML path is hardcoded and lacks cache/versioning
- **Category:** usability / maintainability
- **Symptoms:** `dashboard_server._load_dashboard_html` reads a static file from `src/esper/karn/dashboard.html`. There’s no versioning or cache-busting, and the server returns a fallback inline HTML on missing file. Changes may not propagate to clients (browser cache), and packaging paths can break when installed as a module.
- **Impact:** Medium – users may see stale dashboard UI or 404 fallback; deploying as a package may fail if path differs.
- **Triggers:** Installing as package, changing dashboard.html without cache-bust, running from different CWDs.
- **Root-Cause Hypothesis:** Simple file load approach without packaging considerations.
- **Remediation Options:** 
  - A) Serve dashboard as a packaged resource with version hash and cache-control headers.
  - B) Add ETag or cache-busting query to WebSocket/HTML load.
  - C) Log when fallback HTML is served to catch missing file issues.
- **Validation Plan:** Package install test; ensure dashboard loads with correct content and updates after changes.
- **Status:** Closed (Fixed)
- **Resolution:** Fixed by adding warning logs when the dashboard HTML is missing and serving the response with `Cache-Control: no-cache` headers to prevent stale UI issues.
- **Links:** `src/esper/karn/dashboard_server.py::_load_dashboard_html`
