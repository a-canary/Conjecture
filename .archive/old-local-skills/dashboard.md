# /dashboard — Project Dashboard Generator

Generate a dashboard manifest for this project. Analyze the codebase, tests, builds,
and any domain-specific metrics to produce a comprehensive status page.

## Output

Write to `/data/dashboard/manifest.json` following this schema:

```json
{
  "project": "<project-name>",
  "generated_at": "<ISO 8601 timestamp>",
  "health": "green|yellow|red",
  "summary": "<one-line status summary>",
  "pages": {
    "overview": {
      "title": "Overview",
      "icon": "home",
      "blocks": [ ... ]
    }
  }
}
```

## Block Types

Use these block types in page `blocks` arrays:

- **text**: `{"type": "text", "content": "## Markdown content here..."}`
- **chart**: `{"type": "chart", "title": "...", "chart_type": "line|bar|pie|doughnut", "data": {"labels": [], "datasets": [{"label": "...", "data": []}]}}`
- **log**: `{"type": "log", "title": "...", "lines": ["line1", "line2"]}`
- **table**: `{"type": "table", "title": "...", "columns": ["Col1", "Col2"], "rows": [["val1", "val2"]]}`
- **image**: `{"type": "image", "title": "...", "src": "filename.png"}` (relative to `assets/`)
- **video**: `{"type": "video", "title": "...", "src": "filename.mp4"}` (relative to `assets/`)
- **links**: `{"type": "links", "items": [{"label": "Page Name", "page": "page-key"}]}`

## Steps

1. **Gather data**: Run git log, git status, check test results, check build status, look for domain-specific metrics (logs, data files, reports).

2. **Determine health**:
   - `green`: tests pass, no uncommitted changes, build clean
   - `yellow`: minor issues (uncommitted changes, skipped tests, warnings)
   - `red`: failing tests, broken build, blocked work

3. **Build overview page**: Always create an `overview` page with:
   - A text block summarizing project status
   - A table of recent commits (last 10)
   - A log block with recent activity

4. **Build domain pages**: Add additional pages relevant to the project. Examples:
   - Trading project: positions, P&L charts, trade log
   - ML project: training metrics, model performance, data pipeline status
   - Web app: deployment status, error rates, user metrics

5. **Export assets**: If you generate charts as images or find relevant visual assets, copy them to `/data/dashboard/assets/`. Reference them in image blocks with just the filename.

6. **Write manifest**: Write the complete JSON to `/data/dashboard/manifest.json`. Ensure it's valid JSON.

7. **Self-improve**: After generating the dashboard, review this skill file. If you found new data sources or better ways to present the project's status, update this skill file at `/workspace/.claude/skills/dashboard.md` for next time.

## Guidelines

- Be terse. Summaries should be one line.
- Use markdown in text blocks for structure.
- Chart data should be actual numbers from the project, not placeholders.
- Log blocks: most recent entries first, limit to 50 lines.
- Table blocks: limit to 100 rows.
- Health should reflect actual project state, not aspirations.
- Always include `generated_at` as a UTC ISO 8601 timestamp.
