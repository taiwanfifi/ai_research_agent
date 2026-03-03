# Research AI — Visualization Dashboard

Read-only visualization for missions data. Zero external dependencies (Python stdlib + vanilla JS + D3 CDN).

## Web Dashboard

```bash
# From the ai_research_agent/ directory:
python3 visual/server.py

# Custom port / missions path:
python3 visual/server.py --port 9090 --missions /path/to/missions
```

Open http://localhost:8080 in your browser.

**Features:** Mission list with search/sort, timeline view, DAG force graph (D3.js), knowledge tree, code version browser with diffs, report viewer.

**Note:** D3.js is loaded from CDN (`d3js.org`). After first load it will be cached by the browser.

## Terminal CLI

```bash
# List all missions
python3 visual/cli.py

# Mission overview
python3 visual/cli.py -m <mission_id_or_slug>

# Specific views
python3 visual/cli.py -m <id> --tasks
python3 visual/cli.py -m <id> --insights
python3 visual/cli.py -m <id> --code
python3 visual/cli.py -m <id> --knowledge
python3 visual/cli.py -m <id> --reports

# Auto-refresh mode
python3 visual/cli.py --watch
python3 visual/cli.py -m <id> --tasks --watch
```

## API Endpoints

| Route | Description |
|-------|-------------|
| `GET /api/missions` | List all missions |
| `GET /api/mission/{id}` | Mission manifest + checkpoint |
| `GET /api/mission/{id}/insights` | InsightDAG (or synthesized) |
| `GET /api/mission/{id}/code` | Code versions + module maps |
| `GET /api/mission/{id}/code/{file}/diff/{v1}/{v2}` | Diff content |
| `GET /api/mission/{id}/knowledge` | Knowledge categories |
| `GET /api/mission/{id}/reports` | Report list |
| `GET /api/mission/{id}/reports/{filename}` | Report content |
| `GET /api/mission/{id}/timeline` | Checkpoint history |
