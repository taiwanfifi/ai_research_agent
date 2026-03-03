/* ================================================================
   Research AI Dashboard — app.js
   Vanilla JS: State, API, UI, DAG, Events, Utils
   ================================================================ */

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------
const Utils = {
    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    },

    /** Gray → deep blue gradient based on 0-1 relevance */
    relevanceColor(r) {
        const clamped = Math.max(0, Math.min(1, r || 0));
        const gray = Math.round(200 - clamped * 160);
        const blue = Math.round(50 + clamped * 150);
        return `rgb(${gray - 60}, ${gray - 20}, ${blue + 50})`;
    },

    relevanceBarHtml(r, blocks = 10) {
        const filled = Math.round((r || 0) * blocks);
        let html = '<span class="relevance-bar">';
        for (let i = 0; i < blocks; i++) {
            const color = i < filled ? Utils.relevanceColor((i + 1) / blocks) : '#E5E7EB';
            html += `<span class="block" style="background:${color}"></span>`;
        }
        html += '</span>';
        return html;
    },

    workerColor(worker) {
        const colors = { explorer: '#2563EB', coder: '#059669', reviewer: '#D97706' };
        return colors[worker] || '#6B7280';
    },

    /** Markdown → HTML (headings, bold, italic, code, links, lists, tables) */
    simpleMarkdown(md) {
        if (!md) return '';
        // Strip <think>...</think> blocks from LLM responses
        let cleaned = md.replace(/<think>[\s\S]*?<\/think>\s*/g, '');
        let html = Utils.escapeHtml(cleaned);

        // Code blocks (fenced)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>');
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Headers
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        // Bold / italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // Horizontal rules
        html = html.replace(/^---$/gm, '<hr>');

        // Tables: detect lines with | delimiters
        html = html.replace(/((?:^.*\|.*$\n?)+)/gm, (match) => {
            const rows = match.trim().split('\n').filter(r => r.trim());
            if (rows.length < 2) return match;
            // Skip separator row (---|---|---)
            const isHeaderSep = (r) => /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?\s*$/.test(r);
            let headerRow = null;
            const dataRows = [];
            for (let i = 0; i < rows.length; i++) {
                if (isHeaderSep(rows[i])) continue;
                const cells = rows[i].split('|').map(c => c.trim()).filter(c => c !== '');
                if (!headerRow) headerRow = cells;
                else dataRows.push(cells);
            }
            if (!headerRow) return match;
            let table = '<table class="md-table"><thead><tr>';
            headerRow.forEach(h => { table += `<th>${h}</th>`; });
            table += '</tr></thead><tbody>';
            dataRows.forEach(row => {
                table += '<tr>';
                row.forEach(c => { table += `<td>${c}</td>`; });
                table += '</tr>';
            });
            table += '</tbody></table>';
            return table;
        });

        // Unordered lists
        html = html.replace(/((?:^[-*] .+$\n?)+)/gm, (match) => {
            const items = match.trim().split('\n')
                .filter(l => l.trim())
                .map(l => `<li>${l.replace(/^[-*] /, '')}</li>`)
                .join('');
            return `<ul>${items}</ul>`;
        });

        // Numbered lists
        html = html.replace(/((?:^\d+\. .+$\n?)+)/gm, (match) => {
            const items = match.trim().split('\n')
                .filter(l => l.trim())
                .map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`)
                .join('');
            return `<ol>${items}</ol>`;
        });

        // Line breaks → paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        html = '<p>' + html + '</p>';
        // Clean up empty paragraphs around block elements
        html = html.replace(/<p>\s*(<(?:h[1-4]|hr|pre|table|ul|ol))/g, '$1');
        html = html.replace(/(<\/(?:h[1-4]|hr|pre|table|ul|ol)>)\s*<\/p>/g, '$1');
        return html;
    },

    formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        return (bytes / 1024).toFixed(1) + ' KB';
    },

    formatDuration(seconds) {
        if (!seconds) return '-';
        if (seconds < 60) return seconds.toFixed(1) + 's';
        const m = Math.floor(seconds / 60);
        const s = Math.round(seconds % 60);
        return `${m}m ${s}s`;
    },

    formatDate(iso) {
        if (!iso) return '';
        return iso.replace('T', ' ').slice(0, 16);
    },

    /** Diff text → colored HTML */
    diffHtml(text) {
        if (!text) return '<span class="empty-state">No diff content</span>';
        return text.split('\n').map(line => {
            if (line.startsWith('@@')) return `<span class="diff-hunk">${Utils.escapeHtml(line)}</span>`;
            if (line.startsWith('+')) return `<span class="diff-add">${Utils.escapeHtml(line)}</span>`;
            if (line.startsWith('-')) return `<span class="diff-del">${Utils.escapeHtml(line)}</span>`;
            return Utils.escapeHtml(line);
        }).join('\n');
    },

    moduleChangeClass(mod) {
        if (mod.startsWith('+')) return 'added';
        if (mod.startsWith('~')) return 'modified';
        if (mod.startsWith('-')) return 'removed';
        return '';
    },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const State = {
    missions: [],
    selectedMission: null,

    // Per-mission cached data
    missionDetail: null,
    insights: null,
    code: null,
    knowledge: null,
    reports: null,
    timeline: null,
    workspaceFiles: null,

    activeTab: 'timeline',
    autoRefreshInterval: null,
    filters: {
        timelineWorker: 'all',
        timelineShowFailures: true,
        dagWorker: 'all',
        dagShowArchived: false,
        dagRelevanceMin: 0,
    },
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
const API = {
    async fetchJson(url) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) {
            console.error('API error:', url, e);
            return null;
        }
    },

    missions()                  { return this.fetchJson('/api/missions'); },
    missionDetail(id)           { return this.fetchJson(`/api/mission/${id}`); },
    insights(id)                { return this.fetchJson(`/api/mission/${id}/insights`); },
    code(id)                    { return this.fetchJson(`/api/mission/${id}/code`); },
    knowledge(id)               { return this.fetchJson(`/api/mission/${id}/knowledge`); },
    reports(id)                 { return this.fetchJson(`/api/mission/${id}/reports`); },
    reportContent(id, filename) { return this.fetchJson(`/api/mission/${id}/reports/${filename}`); },
    timeline(id)                { return this.fetchJson(`/api/mission/${id}/timeline`); },
    diff(id, stem, v1, v2)     { return this.fetchJson(`/api/mission/${id}/code/${stem}/diff/${v1}/${v2}`); },
    workspaceFiles(id)          { return this.fetchJson(`/api/mission/${id}/workspace`); },
    workspaceFile(id, path)     { return this.fetchJson(`/api/mission/${id}/workspace/${path}`); },
};

// ---------------------------------------------------------------------------
// UI Renderers
// ---------------------------------------------------------------------------
const UI = {

    // ---- Mission List ----
    renderMissionList() {
        const container = document.getElementById('mission-list');
        let missions = [...State.missions];
        const search = (document.getElementById('mission-search').value || '').toLowerCase();
        if (search) {
            missions = missions.filter(m =>
                (m.goal || '').toLowerCase().includes(search) ||
                (m.slug || '').toLowerCase().includes(search) ||
                (m.id || '').toLowerCase().includes(search)
            );
        }
        const sort = document.getElementById('mission-sort').value;
        if (sort === 'name') missions.sort((a, b) => (a.slug || '').localeCompare(b.slug || ''));
        else if (sort === 'status') missions.sort((a, b) => (a.status || '').localeCompare(b.status || ''));
        else missions.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

        if (!missions.length) {
            container.innerHTML = '<div class="empty-state">No missions found</div>';
            return;
        }

        container.innerHTML = missions.map(m => `
            <div class="mission-card ${State.selectedMission === m.id ? 'selected' : ''}"
                 data-id="${Utils.escapeHtml(m.id)}">
                <div class="mc-title">${Utils.escapeHtml(m.goal || m.slug || m.id)}</div>
                <div class="mc-meta">
                    <span class="status-badge ${m.status || ''}">${Utils.escapeHtml(m.status || 'unknown')}</span>
                    <span>${m.cycle || 0}/${m.max_cycles || 0} cycles</span>
                    <span>${m.task_count || 0} tasks</span>
                </div>
            </div>
        `).join('');
    },

    // ---- Top Bar ----
    renderTopBar() {
        const nameEl = document.getElementById('topbar-mission');
        const statusEl = document.getElementById('topbar-status');
        const progressEl = document.getElementById('topbar-progress');

        if (!State.missionDetail) {
            nameEl.textContent = 'Select a mission';
            statusEl.hidden = true;
            progressEl.hidden = true;
            return;
        }

        const m = State.missionDetail.manifest || {};
        const cp = State.missionDetail.checkpoint || {};
        nameEl.textContent = m.goal || m.slug || m.mission_id || '';
        statusEl.textContent = m.status || cp.state || '';
        statusEl.className = `status-badge ${m.status || cp.state || ''}`;
        statusEl.hidden = false;

        const cycle = cp.cycle || 0;
        const max = cp.max_cycles || 1;
        document.getElementById('progress-label').textContent = `${cycle}/${max}`;
        document.getElementById('progress-fill').style.width = `${Math.round(cycle / max * 100)}%`;
        progressEl.hidden = false;
    },

    // ---- Timeline ----
    renderTimeline() {
        const container = document.getElementById('timeline-content');
        if (!State.missionDetail) {
            container.innerHTML = '<div class="empty-state">Select a mission to view timeline</div>';
            return;
        }

        const cp = State.missionDetail.checkpoint || {};
        let tasks = cp.completed_tasks || [];
        const wf = State.filters.timelineWorker;
        if (wf !== 'all') tasks = tasks.filter(t => t.worker === wf);
        if (!State.filters.timelineShowFailures) tasks = tasks.filter(t => t.success !== false);

        if (!tasks.length) {
            container.innerHTML = '<div class="empty-state">No tasks match filters</div>';
            return;
        }

        container.innerHTML = tasks.map((t, i) => `
            <div class="timeline-item ${t.success === false ? 'failure' : ''}" data-index="${i}">
                <div class="tl-marker">
                    <div class="tl-cycle ${t.worker || ''}">${i + 1}</div>
                </div>
                <div class="tl-body">
                    <div class="tl-task">${Utils.escapeHtml((t.task || '').slice(0, 150))}</div>
                    <div class="tl-meta">
                        <span class="worker-badge ${t.worker || ''}">${Utils.escapeHtml(t.worker || '')}</span>
                        <span>${t.success === false ? 'Failed' : 'OK'}</span>
                        <span>${Utils.formatDuration(t.elapsed_s)}</span>
                    </div>
                </div>
            </div>
        `).join('');
    },

    // ---- Knowledge ----
    renderKnowledge() {
        const container = document.getElementById('knowledge-content');
        if (!State.knowledge || !Object.keys(State.knowledge).length) {
            container.innerHTML = '<div class="empty-state">No knowledge data</div>';
            return;
        }

        const categoryIcons = { papers: '\u{1F4C4}', experiments: '\u{1F52C}', methods: '\u{2699}', code: '\u{1F4BB}', reports: '\u{1F4CA}' };

        container.innerHTML = Object.entries(State.knowledge).map(([cat, data]) => {
            const items = data.items || {};
            const count = data.item_count || Object.keys(items).length;
            const icon = categoryIcons[cat] || '';

            const itemsHtml = Object.entries(items).map(([id, item]) => `
                <div class="knowledge-item" data-category="${Utils.escapeHtml(cat)}" data-id="${Utils.escapeHtml(id)}">
                    <div class="ki-title">${Utils.escapeHtml((item.title || id).slice(0, 100))}</div>
                    <div class="ki-summary">${Utils.escapeHtml(item.summary || '')}</div>
                    <div class="ki-keywords">
                        ${(item.keywords || []).map(k => `<span class="keyword-tag">${Utils.escapeHtml(k)}</span>`).join('')}
                    </div>
                </div>
            `).join('');

            return `
                <div class="knowledge-category">
                    <div class="knowledge-category-header">
                        <span>${icon}</span>
                        <span>${Utils.escapeHtml(cat)}</span>
                        <span class="count">(${count})</span>
                    </div>
                    ${itemsHtml || '<div class="empty-state" style="min-height:40px">Empty</div>'}
                </div>
            `;
        }).join('');
    },

    // ---- Code ----
    renderCode() {
        const container = document.getElementById('code-content');
        let html = '';

        // Workspace files section
        if (State.workspaceFiles && State.workspaceFiles.length) {
            const pyFiles = State.workspaceFiles.filter(f => f.type === 'code');
            const imgFiles = State.workspaceFiles.filter(f => f.type === 'image');
            const dataFiles = State.workspaceFiles.filter(f => f.type === 'data' || f.type === 'other');

            html += '<div class="knowledge-category"><div class="knowledge-category-header">Workspace Files</div>';

            if (pyFiles.length) {
                html += pyFiles.map(f => `
                    <div class="report-item" data-ws-path="${Utils.escapeHtml(f.path)}" data-ws-type="text">
                        <span class="ri-name">${Utils.escapeHtml(f.name)}</span>
                        <span class="ri-size">${Utils.formatBytes(f.size)}</span>
                    </div>
                `).join('');
            }
            if (imgFiles.length) {
                html += '<div style="margin-top:8px">';
                html += imgFiles.map(f => `
                    <div class="report-item" data-ws-path="${Utils.escapeHtml(f.path)}" data-ws-type="image">
                        <span class="ri-name">${Utils.escapeHtml(f.name)}</span>
                        <span class="ri-size">${Utils.formatBytes(f.size)}</span>
                    </div>
                `).join('');
                html += '</div>';
            }
            if (dataFiles.length) {
                html += dataFiles.map(f => `
                    <div class="report-item" data-ws-path="${Utils.escapeHtml(f.path)}" data-ws-type="text">
                        <span class="ri-name">${Utils.escapeHtml(f.name)}</span>
                        <span class="ri-size">${Utils.formatBytes(f.size)}</span>
                    </div>
                `).join('');
            }
            html += '</div>';
        }

        // Code store section
        if (State.code && State.code.length) {
            html += '<div class="knowledge-category" style="margin-top:16px"><div class="knowledge-category-header">Version-Tracked Code</div></div>';
            html += State.code.map(file => {
            const manifest = file.manifest || {};
            const versions = manifest.versions || [];
            const moduleMap = file.module_map || [];

            const versionsHtml = versions.map((v, i) => {
                const mods = (v.modules_changed || []).map(m =>
                    `<span class="module-change ${Utils.moduleChangeClass(m)}">${Utils.escapeHtml(m)}</span>`
                ).join('');
                return `
                    <div class="version-entry">
                        <span class="version-chip" data-stem="${Utils.escapeHtml(file.stem)}"
                              data-version="${Utils.escapeHtml(v.version)}"
                              data-prev="${i > 0 ? Utils.escapeHtml(versions[i-1].version) : ''}">${Utils.escapeHtml(v.version)}</span>
                        <span class="version-reason">${Utils.escapeHtml(v.reason || '')}</span>
                        <div class="version-modules">${mods}</div>
                    </div>
                `;
            }).join('');

            const moduleMapHtml = moduleMap.length ? `
                <div class="module-map">
                    <div class="module-map-title">Module Map</div>
                    ${moduleMap.map(m => `
                        <div class="module-entry">
                            <span class="module-kind">${Utils.escapeHtml(m.kind || '')}</span>
                            <span>${Utils.escapeHtml(m.name || '')}</span>
                            <span style="color:var(--text-muted);font-size:10px">${Utils.escapeHtml(m.signature || '')}</span>
                        </div>
                    `).join('')}
                </div>
            ` : '';

            return `
                <div class="code-file">
                    <div class="code-file-header">
                        <span>${Utils.escapeHtml(manifest.filename || file.stem)}</span>
                        <span class="version-tag">${Utils.escapeHtml(manifest.latest || '')}</span>
                    </div>
                    <div class="version-list">${versionsHtml || '<div class="empty-state" style="min-height:30px">No versions</div>'}</div>
                    ${moduleMapHtml}
                </div>
            `;
        }).join('');
        }

        if (!html) {
            html = '<div class="empty-state">No code or workspace files</div>';
        }
        container.innerHTML = html;
    },

    // ---- Reports ----
    renderReports() {
        const container = document.getElementById('reports-content');
        if (!State.reports || !State.reports.length) {
            container.innerHTML = '<div class="empty-state">No reports</div>';
            return;
        }

        container.innerHTML = State.reports.map(r => `
            <div class="report-item" data-filename="${Utils.escapeHtml(r.filename)}">
                <span class="ri-name">${Utils.escapeHtml(r.filename)}</span>
                <span class="ri-size">${Utils.formatBytes(r.size || 0)}</span>
            </div>
        `).join('');
    },

    // ---- Detail Panel ----
    showDetail(title, sections) {
        document.getElementById('detail-title').textContent = title;
        const body = document.getElementById('detail-body');
        body.innerHTML = sections.map(s => `
            <div class="detail-section">
                <div class="detail-section-title">${Utils.escapeHtml(s.title)}</div>
                <div class="detail-content">${s.html}</div>
            </div>
        `).join('');
        // Ensure detail panel is visible
        const panel = document.getElementById('detail-panel');
        if (panel) panel.classList.add('has-content');

        // Set Q&A context from detail content
        const plainText = sections.map(s => `${s.title}: ${s.html.replace(/<[^>]*>/g, '')}`).join('\n\n');
        QA.setContext(plainText);
    },

    clearDetail() {
        document.getElementById('detail-title').textContent = 'Details';
        document.getElementById('detail-body').innerHTML = '<div class="empty-state">Click any item to see details</div>';
        const panel = document.getElementById('detail-panel');
        if (panel) panel.classList.remove('has-content', 'expanded');
    },

    toggleDetailExpand() {
        const panel = document.getElementById('detail-panel');
        if (panel) panel.classList.toggle('expanded');
    },

    // ---- Tab switching ----
    switchTab(tab) {
        State.activeTab = tab;
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
        document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));
    },
};

// ---------------------------------------------------------------------------
// DAG — D3 Force-Directed Graph
// ---------------------------------------------------------------------------
const DAG = {
    svg: null,
    simulation: null,

    render() {
        const container = document.getElementById('dag-content');
        if (!State.insights || !State.insights.nodes || !Object.keys(State.insights.nodes).length) {
            container.innerHTML = '<div class="empty-state">No insight data</div>';
            return;
        }

        // Filter nodes
        let nodes = Object.values(State.insights.nodes);
        const wf = State.filters.dagWorker;
        if (wf !== 'all') nodes = nodes.filter(n => n.worker === wf);
        if (!State.filters.dagShowArchived) nodes = nodes.filter(n => !n.archived);
        nodes = nodes.filter(n => (n.relevance || 0) >= State.filters.dagRelevanceMin);

        const nodeIds = new Set(nodes.map(n => n.id));

        // Build edges from references
        const links = [];
        nodes.forEach(n => {
            (n.references || []).forEach(ref => {
                if (nodeIds.has(ref)) {
                    links.push({ source: ref, target: n.id });
                }
            });
        });

        // Clear previous
        container.innerHTML = '';

        const width = container.clientWidth || 600;
        const height = container.clientHeight || 400;

        const svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height);

        // Zoom
        const g = svg.append('g');
        svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('class', 'dag-link-arrow');

        // Links
        const link = g.selectAll('.dag-link')
            .data(links)
            .join('line')
            .attr('class', 'dag-link')
            .attr('marker-end', 'url(#arrowhead)');

        // Nodes
        const node = g.selectAll('.dag-node')
            .data(nodes, d => d.id)
            .join('g')
            .attr('class', 'dag-node')
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) DAG.simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
                .on('end', (event, d) => {
                    if (!event.active) DAG.simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                })
            );

        node.append('circle')
            .attr('r', d => 6 + (d.relevance || 0) * 10)
            .attr('fill', d => Utils.workerColor(d.worker))
            .attr('stroke', d => d.success === false ? '#EF4444' : '#fff')
            .attr('stroke-width', d => d.success === false ? 2.5 : 1.5)
            .attr('opacity', d => 0.5 + (d.relevance || 0) * 0.5);

        node.append('text')
            .attr('dy', d => -(10 + (d.relevance || 0) * 10))
            .attr('text-anchor', 'middle')
            .text(d => (d.task || d.id).slice(0, 20));

        // Click handler
        node.on('click', (event, d) => {
            event.stopPropagation();
            const sections = [
                { title: 'Task', html: Utils.escapeHtml(d.task || '') },
                { title: 'Worker', html: `<span class="worker-badge ${d.worker || ''}">${Utils.escapeHtml(d.worker || '')}</span> &middot; Cycle ${d.cycle || '?'} &middot; ${d.success === false ? '<span style="color:var(--status-failure)">Failed</span>' : 'Success'}` },
                { title: 'Relevance', html: `${Utils.relevanceBarHtml(d.relevance)} ${(d.relevance || 0).toFixed(2)}` },
                { title: 'Content', html: `<div class="md-content">${Utils.simpleMarkdown(d.content || 'No content')}</div>` },
            ];
            if (d.references && d.references.length) {
                sections.push({ title: 'References', html: d.references.map(r => `<code>${Utils.escapeHtml(r)}</code>`).join(', ') });
            }
            if (d.code_refs && d.code_refs.length) {
                sections.push({
                    title: 'Code References',
                    html: d.code_refs.map(cr =>
                        `<code>${Utils.escapeHtml(cr.filename || '')}@${Utils.escapeHtml(cr.version || '')}</code>` +
                        (cr.modules_changed ? ` (${cr.modules_changed.map(m => `<span class="module-change ${Utils.moduleChangeClass(m)}">${Utils.escapeHtml(m)}</span>`).join(' ')})` : '')
                    ).join('<br>')
                });
            }
            if (d.synthetic) {
                sections.push({ title: 'Note', html: '<em>Synthesized from completed tasks (legacy mission without InsightDAG)</em>' });
            }
            UI.showDetail(`Insight ${d.id}`, sections);
        });

        // Simulation
        DAG.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => 10 + (d.relevance || 0) * 12))
            .on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });

        DAG.svg = svg;
    },
};

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
const Events = {
    init() {
        // Tab clicks
        document.getElementById('tab-bar').addEventListener('click', e => {
            const tab = e.target.closest('.tab');
            if (tab) UI.switchTab(tab.dataset.tab);
        });

        // Mission list clicks
        document.getElementById('mission-list').addEventListener('click', e => {
            const card = e.target.closest('.mission-card');
            if (card) Events.selectMission(card.dataset.id);
        });

        // Search & sort
        document.getElementById('mission-search').addEventListener('input', () => UI.renderMissionList());
        document.getElementById('mission-sort').addEventListener('change', () => UI.renderMissionList());

        // Timeline filters
        document.getElementById('timeline-worker-filter').addEventListener('change', e => {
            State.filters.timelineWorker = e.target.value;
            UI.renderTimeline();
        });
        document.getElementById('timeline-show-failures').addEventListener('change', e => {
            State.filters.timelineShowFailures = e.target.checked;
            UI.renderTimeline();
        });

        // DAG filters
        document.getElementById('dag-worker-filter').addEventListener('change', e => {
            State.filters.dagWorker = e.target.value;
            DAG.render();
        });
        document.getElementById('dag-show-archived').addEventListener('change', e => {
            State.filters.dagShowArchived = e.target.checked;
            DAG.render();
        });
        document.getElementById('dag-relevance-slider').addEventListener('input', e => {
            State.filters.dagRelevanceMin = parseFloat(e.target.value);
            document.getElementById('dag-relevance-value').textContent = State.filters.dagRelevanceMin.toFixed(2);
            DAG.render();
        });

        // Timeline item clicks
        document.getElementById('timeline-content').addEventListener('click', e => {
            const item = e.target.closest('.timeline-item');
            if (!item) return;
            const idx = parseInt(item.dataset.index);
            const tasks = (State.missionDetail?.checkpoint?.completed_tasks) || [];
            const t = tasks[idx];
            if (!t) return;
            document.querySelectorAll('.timeline-item').forEach(el => el.classList.remove('selected'));
            item.classList.add('selected');
            UI.showDetail(`Task #${idx + 1}`, [
                { title: 'Task', html: Utils.escapeHtml(t.task || '') },
                { title: 'Worker', html: `<span class="worker-badge ${t.worker || ''}">${Utils.escapeHtml(t.worker || '')}</span>` },
                { title: 'Status', html: t.success === false ? '<span style="color:var(--status-failure)">Failed</span>' : '<span style="color:var(--status-success)">Success</span>' },
                { title: 'Duration', html: Utils.formatDuration(t.elapsed_s) },
                { title: 'Output', html: `<div class="md-content">${Utils.simpleMarkdown(t.output || 'No output')}</div>` },
                ...(t.error ? [{ title: 'Error', html: `<pre style="color:var(--status-failure)">${Utils.escapeHtml(t.error)}</pre>` }] : []),
            ]);
        });

        // Knowledge item clicks
        document.getElementById('knowledge-content').addEventListener('click', e => {
            const item = e.target.closest('.knowledge-item');
            if (!item) return;
            const cat = item.dataset.category;
            const id = item.dataset.id;
            const catData = State.knowledge?.[cat];
            const itemData = catData?.items?.[id];
            if (!itemData) return;
            UI.showDetail(itemData.title || id, [
                { title: 'Category', html: Utils.escapeHtml(cat) },
                { title: 'Title', html: Utils.escapeHtml(itemData.title || '') },
                { title: 'Summary', html: Utils.escapeHtml(itemData.summary || '') },
                { title: 'Keywords', html: (itemData.keywords || []).map(k => `<span class="keyword-tag">${Utils.escapeHtml(k)}</span>`).join(' ') },
                { title: 'File', html: `<code>${Utils.escapeHtml(itemData.file || '')}</code> (${Utils.formatBytes(itemData.size || 0)})` },
            ]);
        });

        // Code version chip clicks → load diff
        document.getElementById('code-content').addEventListener('click', async e => {
            const chip = e.target.closest('.version-chip');
            if (!chip) return;
            const stem = chip.dataset.stem;
            const version = chip.dataset.version;
            const prev = chip.dataset.prev;

            document.querySelectorAll('.version-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            if (!prev) {
                UI.showDetail(`${stem} ${version}`, [
                    { title: 'Version', html: `<code>${Utils.escapeHtml(version)}</code> (initial version)` },
                ]);
                return;
            }

            UI.showDetail(`${stem} ${prev} → ${version}`, [
                { title: 'Loading', html: 'Fetching diff...' },
            ]);

            const data = await API.diff(State.selectedMission, stem, prev, version);
            if (data && data.diff) {
                UI.showDetail(`${stem} ${prev} → ${version}`, [
                    { title: 'Diff', html: `<div class="diff-view">${Utils.diffHtml(data.diff)}</div>` },
                ]);
            } else {
                UI.showDetail(`${stem} ${prev} → ${version}`, [
                    { title: 'Diff', html: '<span class="empty-state">Diff not available</span>' },
                ]);
            }
        });

        // Report item clicks
        document.getElementById('reports-content').addEventListener('click', async e => {
            const item = e.target.closest('.report-item');
            if (!item) return;
            const filename = item.dataset.filename;
            UI.showDetail(filename, [{ title: 'Loading', html: 'Fetching report...' }]);

            const data = await API.reportContent(State.selectedMission, filename);
            if (data && data.content) {
                UI.showDetail(filename, [
                    { title: 'Report', html: `<div class="md-content">${Utils.simpleMarkdown(data.content)}</div>` },
                ]);
            } else {
                UI.showDetail(filename, [
                    { title: 'Report', html: '<span class="empty-state">Could not load report</span>' },
                ]);
            }
        });

        // Workspace file clicks
        document.getElementById('code-content').addEventListener('click', async e => {
            // Handle workspace file clicks
            const wsItem = e.target.closest('[data-ws-path]');
            if (wsItem) {
                const wsPath = wsItem.dataset.wsPath;
                const wsType = wsItem.dataset.wsType;
                if (wsType === 'image') {
                    UI.showDetail(wsPath.split('/').pop(), [
                        { title: 'Image', html: `<img src="/api/mission/${State.selectedMission}/workspace/${wsPath}" style="max-width:100%;border-radius:4px">` },
                    ]);
                } else {
                    UI.showDetail(wsPath.split('/').pop(), [{ title: 'Loading', html: 'Fetching...' }]);
                    const data = await API.workspaceFile(State.selectedMission, wsPath);
                    if (data && data.content) {
                        const ext = wsPath.split('.').pop();
                        const contentHtml = ext === 'py' ?
                            `<pre>${Utils.escapeHtml(data.content)}</pre>` :
                            ext === 'md' ?
                            `<div class="md-content">${Utils.simpleMarkdown(data.content)}</div>` :
                            `<pre>${Utils.escapeHtml(data.content)}</pre>`;
                        UI.showDetail(wsPath.split('/').pop(), [
                            { title: wsPath, html: contentHtml },
                        ]);
                    }
                }
                return;
            }
        });

        // Detail close & expand
        document.getElementById('detail-close').addEventListener('click', () => UI.clearDetail());
        const expandBtn = document.getElementById('detail-expand');
        if (expandBtn) expandBtn.addEventListener('click', () => UI.toggleDetailExpand());
    },

    async selectMission(id) {
        State.selectedMission = id;
        UI.renderMissionList();
        UI.clearDetail();

        // Parallel fetch all data
        const [detail, insights, code, knowledge, reports, timeline, wsFiles] = await Promise.all([
            API.missionDetail(id),
            API.insights(id),
            API.code(id),
            API.knowledge(id),
            API.reports(id),
            API.timeline(id),
            API.workspaceFiles(id),
        ]);

        State.missionDetail = detail;
        State.insights = insights;
        State.code = code;
        State.knowledge = knowledge;
        State.reports = reports;
        State.timeline = timeline;
        State.workspaceFiles = wsFiles;

        UI.renderTopBar();
        UI.renderTimeline();
        DAG.render();
        UI.renderKnowledge();
        UI.renderCode();
        UI.renderReports();

        // Start auto-refresh for running missions
        Events.startAutoRefresh();
    },

    startAutoRefresh() {
        if (State.autoRefreshInterval) clearInterval(State.autoRefreshInterval);
        const status = State.missionDetail?.manifest?.status;
        if (status === 'running' || status === 'planning') {
            State.autoRefreshInterval = setInterval(async () => {
                if (!State.selectedMission) return;
                const [detail, insights, wsFiles] = await Promise.all([
                    API.missionDetail(State.selectedMission),
                    API.insights(State.selectedMission),
                    API.workspaceFiles(State.selectedMission),
                ]);
                State.missionDetail = detail;
                State.insights = insights;
                State.workspaceFiles = wsFiles;
                UI.renderTopBar();
                UI.renderTimeline();
                UI.renderCode();
                // Refresh mission list too
                State.missions = await API.missions() || [];
                UI.renderMissionList();
            }, 10000); // 10s interval
        }
    },
};

// ---------------------------------------------------------------------------
// Q&A — LLM Chat in Detail Panel
// ---------------------------------------------------------------------------
const QA = {
    currentContext: '',  // The content of whatever is displayed in the detail panel

    setContext(text) {
        this.currentContext = text || '';
        // Clear previous messages when context changes
        document.getElementById('qa-messages').innerHTML = '';
    },

    async ask(question) {
        if (!question.trim()) return;
        const msgContainer = document.getElementById('qa-messages');

        // Show user message
        const userMsg = document.createElement('div');
        userMsg.className = 'qa-msg user';
        userMsg.textContent = question;
        msgContainer.appendChild(userMsg);

        // Show loading
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'qa-msg assistant loading';
        loadingMsg.textContent = 'Thinking...';
        msgContainer.appendChild(loadingMsg);
        msgContainer.scrollTop = msgContainer.scrollHeight;

        try {
            const resp = await fetch('/api/qa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    context: this.currentContext,
                }),
            });
            const data = await resp.json();
            loadingMsg.className = 'qa-msg assistant';
            loadingMsg.innerHTML = `<div class="md-content">${Utils.simpleMarkdown(data.answer || data.error || 'No response')}</div>`;
        } catch (e) {
            loadingMsg.className = 'qa-msg assistant';
            loadingMsg.textContent = `Error: ${e.message}`;
        }
        msgContainer.scrollTop = msgContainer.scrollHeight;
    },

    init() {
        const input = document.getElementById('qa-input');
        const sendBtn = document.getElementById('qa-send');

        const doSend = () => {
            const q = input.value.trim();
            if (q) {
                this.ask(q);
                input.value = '';
            }
        };

        sendBtn.addEventListener('click', doSend);
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') doSend();
        });
    },
};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
(async function init() {
    Events.init();
    QA.init();
    State.missions = await API.missions() || [];
    UI.renderMissionList();

    // Auto-select first mission if available
    if (State.missions.length) {
        Events.selectMission(State.missions[0].id);
    }
})();
