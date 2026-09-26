"""
High-Performance, Zero-Dependency Glassmorphic Web Dashboard.
Runs out of the box using Python's standard library http.server on ANY Windows/Mac/Linux laptop.
No pip packages required!
"""

import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path
import subprocess
import threading
import time
import webbrowser
import os
import sys

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.path_utils import get_environment_info, detect_test_dir, detect_output_dir, detect_cache_dir
from src.progress_state import read_progress_state

PORT = 5000
RUNNER_PROCESS = None
RUNNER_LOGS = []
RUNNER_LOCK = threading.Lock()
ACTIVE_SHARD_ID = 0
ACTIVE_NUM_SHARDS = 3


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Entity Resolution - Distributed Inference Hub</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #07090e;
            --card-bg: rgba(18, 23, 37, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #7f00ff;
            --accent-emerald: #00f5a0;
            --accent-amber: #f6d365;
            --accent-rose: #ff0844;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --glass-blur: blur(20px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(79, 172, 254, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(127, 0, 255, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(0, 245, 160, 0.05) 0%, transparent 60%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1.5rem 4rem 1.5rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
        }

        .brand-title {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00f5a0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-sub {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .sys-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.85rem;
            color: var(--accent-emerald);
        }

        .dot-pulse {
            width: 8px;
            height: 8px;
            background-color: var(--accent-emerald);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
        }

        /* Environment Cards */
        .env-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .env-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: var(--glass-blur);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            transition: all 0.25s ease;
        }

        .env-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .env-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.35rem;
        }

        .env-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-main);
            word-break: break-all;
        }

        .tag-pill {
            display: inline-block;
            margin-top: 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
        }

        .tag-green { background: rgba(0, 245, 160, 0.15); color: var(--accent-emerald); border: 1px solid rgba(0, 245, 160, 0.3); }
        .tag-amber { background: rgba(246, 211, 101, 0.15); color: var(--accent-amber); border: 1px solid rgba(246, 211, 101, 0.3); }

        /* Shard Selector */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .shard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .shard-option {
            background: var(--card-bg);
            border: 2px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem 1.25rem;
            cursor: pointer;
            position: relative;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            backdrop-filter: var(--glass-blur);
        }

        .shard-option:hover {
            border-color: rgba(79, 172, 254, 0.5);
            background: rgba(25, 33, 52, 0.7);
            transform: translateY(-3px);
        }

        .shard-option.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.08);
            box-shadow: 0 0 25px rgba(0, 242, 254, 0.2);
        }

        .shard-badge {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .shard-name {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .shard-range {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Action Buttons */
        .actions-bar {
            display: flex;
            gap: 1rem;
            margin-bottom: 2.5rem;
            flex-wrap: wrap;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            padding: 0.9rem 2rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #07090e;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35);
        }

        .btn-primary:hover {
            box-shadow: 0 6px 28px rgba(0, 242, 254, 0.55);
            transform: translateY(-2px);
        }

        .btn-danger {
            background: rgba(255, 8, 68, 0.15);
            color: #ff3366;
            border: 1px solid rgba(255, 8, 68, 0.3);
        }

        .btn-danger:hover {
            background: rgba(255, 8, 68, 0.25);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-main);
            border: 1px solid var(--card-border);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* Live Progress Section */
        .progress-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: var(--glass-blur);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 1.25rem;
        }

        .progress-stage {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .progress-pct {
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            background: linear-gradient(135deg, #00f2fe 0%, #00f5a0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .bar-container {
            width: 100%;
            height: 14px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #00f2fe, #4facfe, #00f5a0);
            border-radius: 999px;
            transition: width 0.4s ease;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.25rem;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 700;
        }

        /* Console Stream */
        .terminal-card {
            background: #06080d;
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 2rem;
        }

        .terminal-header {
            background: #0f131c;
            padding: 0.75rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
        }

        .term-dots {
            display: flex;
            gap: 6px;
        }

        .term-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .dot-red { background: #ff5f56; }
        .dot-yellow { background: #ffbd2e; }
        .dot-green { background: #27c93f; }

        .term-title {
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .terminal-body {
            height: 280px;
            padding: 1rem;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #d1d5db;
            white-space: pre-wrap;
            scroll-behavior: smooth;
        }

        /* Merge Section */
        .merge-box {
            background: rgba(127, 0, 255, 0.06);
            border: 1px solid rgba(127, 0, 255, 0.25);
            border-radius: 16px;
            padding: 1.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        .merge-info h3 {
            font-size: 1.2rem;
            margin-bottom: 0.4rem;
        }

        .merge-info p {
            font-size: 0.9rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div>
                <div class="brand-title">BUSINESS ENTITY RESOLUTION</div>
                <div class="brand-sub">Distributed Sharded Multi-Laptop Inference Hub</div>
            </div>
            <div class="sys-badge">
                <span class="dot-pulse"></span>
                <span id="sys-cores">8 CPU Cores Active</span>
            </div>
        </header>

        <!-- Auto-Detected Environment Info -->
        <div class="env-bar">
            <div class="env-card">
                <div class="env-label">Test Data Source</div>
                <div class="env-value" id="val-test">Scanning...</div>
                <span class="tag-pill tag-green" id="tag-test">Auto-Detected</span>
            </div>
            <div class="env-card">
                <div class="env-label">Normalization & Blocking Cache</div>
                <div class="env-value" id="val-cache">Scanning...</div>
                <span class="tag-pill tag-green" id="tag-cache">Cache Active</span>
            </div>
            <div class="env-card">
                <div class="env-label">Output Directory</div>
                <div class="env-value" id="val-out">Scanning...</div>
                <span class="tag-pill tag-green">Drive Destination</span>
            </div>
        </div>

        <!-- Shard Selection -->
        <div class="section-title">
            <span>Select Machine / Shard Partition</span>
        </div>
        <div class="shard-grid">
            <div class="shard-option active" onclick="selectShard(0, 3)" id="shard-card-0">
                <div class="shard-badge">LAPTOP 1</div>
                <div class="shard-name">Part 1 of 3</div>
                <div class="shard-range">Entities 0 - 577,514</div>
            </div>
            <div class="shard-option" onclick="selectShard(1, 3)" id="shard-card-1">
                <div class="shard-badge">LAPTOP 2</div>
                <div class="shard-name">Part 2 of 3</div>
                <div class="shard-range">Entities 577,515 - 1,155,029</div>
            </div>
            <div class="shard-option" onclick="selectShard(2, 3)" id="shard-card-2">
                <div class="shard-badge">LAPTOP 3</div>
                <div class="shard-name">Part 3 of 3</div>
                <div class="shard-range">Entities 1,155,030 - 1,732,544</div>
            </div>
            <div class="shard-option" onclick="selectShard(0, 1)" id="shard-card-full">
                <div class="shard-badge">SINGLE MACHINE</div>
                <div class="shard-name">Full Run (100%)</div>
                <div class="shard-range">All 1,732,544 Entities</div>
            </div>
        </div>

        <!-- Action Control Buttons -->
        <div class="actions-bar">
            <button class="btn btn-primary" id="btn-run" onclick="startExecution()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                Start Shard Inference
            </button>
            <button class="btn btn-danger" id="btn-stop" onclick="stopExecution()" style="display:none;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>
                Stop Execution
            </button>
        </div>

        <!-- Live Progress Section -->
        <div class="progress-card">
            <div class="progress-header">
                <div>
                    <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.25rem;">CURRENT PIPELINE STAGE</div>
                    <div class="progress-stage" id="stage-text">Ready to run</div>
                </div>
                <div class="progress-pct" id="pct-text">0.0%</div>
            </div>

            <div class="bar-container">
                <div class="bar-fill" id="progress-bar"></div>
            </div>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Entities Processed</div>
                    <div class="metric-value" id="val-entities">0 / 577,515</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Chunks Completed</div>
                    <div class="metric-value" id="val-chunks">0 / 6</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Processing Speed</div>
                    <div class="metric-value" id="val-speed">0 ent/s</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Time Elapsed</div>
                    <div class="metric-value" id="val-elapsed">00:00</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Estimated Time Left</div>
                    <div class="metric-value" id="val-eta">--:--</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Candidates Found</div>
                    <div class="metric-value" id="val-cands">0</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Matches Confirmed</div>
                    <div class="metric-value" id="val-matches">0</div>
                </div>
            </div>
        </div>

        <!-- Live Terminal Stream -->
        <div class="terminal-card">
            <div class="terminal-header">
                <div class="term-dots">
                    <div class="term-dot dot-red"></div>
                    <div class="term-dot dot-yellow"></div>
                    <div class="term-dot dot-green"></div>
                </div>
                <div class="term-title">EXECUTION CONSOLE OUTPUT</div>
                <div style="font-size:0.75rem; color:var(--text-muted); cursor:pointer;" onclick="document.getElementById('console-box').textContent=''">Clear</div>
            </div>
            <div class="terminal-body" id="console-box">Initializing Dashboard... Ready to start.</div>
        </div>

        <!-- Merger Tool Card -->
        <div class="merge-box">
            <div class="merge-info">
                <h3>Final Step: Merge All Shards</h3>
                <p>When all 3 laptops finish, copy the output files into the output folder and merge with 1 click.</p>
            </div>
            <button class="btn btn-secondary" onclick="triggerMerge()">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
                Merge 3 Shards & Validate
            </button>
        </div>
    </div>

    <script>
        let selectedShardId = 0;
        let selectedNumShards = 3;
        let isRunning = false;

        function selectShard(shardId, numShards) {
            if (isRunning) return;
            selectedShardId = shardId;
            selectedNumShards = numShards;

            document.querySelectorAll('.shard-option').forEach(el => el.classList.remove('active'));
            if (numShards === 1) {
                document.getElementById('shard-card-full').classList.add('active');
            } else {
                document.getElementById('shard-card-' + shardId).classList.add('active');
            }
        }

        function formatTime(seconds) {
            if (isNaN(seconds) || seconds < 0) return "--:--";
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
        }

        async function fetchEnv() {
            try {
                const res = await fetch('/api/env');
                const env = await res.json();
                document.getElementById('val-test').textContent = env.test_dir;
                document.getElementById('val-cache').textContent = env.cache_dir;
                document.getElementById('val-out').textContent = env.output_dir;
                document.getElementById('sys-cores').textContent = env.cpu_count + " CPU Cores Detected";

                if (env.has_norm_cache && env.has_index_cache) {
                    document.getElementById('tag-cache').textContent = "Turbo Mode (Instant Startup)";
                    document.getElementById('tag-cache').className = "tag-pill tag-green";
                } else {
                    document.getElementById('tag-cache').textContent = "Cache will be built on 1st run";
                    document.getElementById('tag-cache').className = "tag-pill tag-amber";
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function startExecution() {
            if (isRunning) return;
            try {
                const res = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ shard_id: selectedShardId, num_shards: selectedNumShards })
                });
                const data = await res.json();
                if (data.status === 'started') {
                    isRunning = true;
                    document.getElementById('btn-run').style.display = 'none';
                    document.getElementById('btn-stop').style.display = 'inline-flex';
                }
            } catch (err) {
                alert("Failed to start: " + err);
            }
        }

        async function stopExecution() {
            if (!confirm("Are you sure you want to stop execution?")) return;
            try {
                await fetch('/api/stop', { method: 'POST' });
                isRunning = false;
                document.getElementById('btn-run').style.display = 'inline-flex';
                document.getElementById('btn-stop').style.display = 'none';
            } catch (err) {
                alert("Failed to stop: " + err);
            }
        }

        async function triggerMerge() {
            if (!confirm("Merge all 3 shards into final candidate_pairs.tsv and matching_results.tsv?")) return;
            try {
                const res = await fetch('/api/merge', { method: 'POST' });
                const data = await res.json();
                alert(data.message);
            } catch (err) {
                alert("Merge error: " + err);
            }
        }

        async function pollProgress() {
            try {
                const res = await fetch('/api/status');
                const s = await res.json();

                isRunning = s.is_running;
                if (isRunning) {
                    document.getElementById('btn-run').style.display = 'none';
                    document.getElementById('btn-stop').style.display = 'inline-flex';
                } else {
                    document.getElementById('btn-run').style.display = 'inline-flex';
                    document.getElementById('btn-stop').style.display = 'none';
                }

                const state = s.state || {};
                document.getElementById('stage-text').textContent = state.stage || (isRunning ? 'Processing...' : 'Ready');
                const pct = state.percent || 0.0;
                document.getElementById('pct-text').textContent = pct.toFixed(1) + "%";
                document.getElementById('progress-bar').style.width = pct + "%";

                if (state.total_entities) {
                    document.getElementById('val-entities').textContent = 
                        (state.entities_processed || 0).toLocaleString() + " / " + (state.total_entities || 0).toLocaleString();
                }
                if (state.total_chunks) {
                    document.getElementById('val-chunks').textContent = 
                        (state.current_chunk || 0) + " / " + state.total_chunks;
                }
                document.getElementById('val-speed').textContent = 
                    Math.round(state.speed_eps || 0).toLocaleString() + " ent/s";
                document.getElementById('val-elapsed').textContent = formatTime(state.elapsed_seconds || 0);
                document.getElementById('val-eta').textContent = formatTime(state.eta_seconds || 0);
                document.getElementById('val-cands').textContent = (state.candidates_count || 0).toLocaleString();
                document.getElementById('val-matches').textContent = (state.matches_count || 0).toLocaleString();

                if (s.logs && s.logs.length > 0) {
                    const cBox = document.getElementById('console-box');
                    cBox.textContent = s.logs.join('\\n');
                    cBox.scrollTop = cBox.scrollHeight;
                }
            } catch (err) {
                console.error(err);
            }
        }

        fetchEnv();
        setInterval(pollProgress, 1000);
    </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/env":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(get_environment_info()).encode("utf-8"))
        elif parsed.path == "/api/status":
            out_d = detect_output_dir()
            state = read_progress_state(out_d)
            global RUNNER_PROCESS, RUNNER_LOGS
            is_running = False
            with RUNNER_LOCK:
                if RUNNER_PROCESS is not None:
                    if RUNNER_PROCESS.poll() is None:
                        is_running = True
                    else:
                        RUNNER_PROCESS = None
                logs = list(RUNNER_LOGS[-100:])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "is_running": is_running,
                "state": state,
                "logs": logs
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        
        if parsed.path == "/api/start":
            params = json.loads(post_body) if post_body else {}
            shard_id = params.get("shard_id", 0)
            num_shards = params.get("num_shards", 3)
            self.start_runner(shard_id, num_shards)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "shard_id": shard_id, "num_shards": num_shards}).encode("utf-8"))

        elif parsed.path == "/api/stop":
            self.stop_runner()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))

        elif parsed.path == "/api/merge":
            res = self.run_merge()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def start_runner(self, shard_id: int, num_shards: int):
        global RUNNER_PROCESS, RUNNER_LOGS, ACTIVE_SHARD_ID, ACTIVE_NUM_SHARDS
        with RUNNER_LOCK:
            if RUNNER_PROCESS is not None and RUNNER_PROCESS.poll() is None:
                return  # already running

            ACTIVE_SHARD_ID = shard_id
            ACTIVE_NUM_SHARDS = num_shards
            RUNNER_LOGS = [f"--- Launching Part {shard_id + 1} of {num_shards} ---"]

            py_exe = sys.executable
            cmd = [
                py_exe,
                str(_PROJECT_DIR / "src" / "shard_infer.py"),
                "--shard-id", str(shard_id),
                "--num-shards", str(num_shards)
            ]

            RUNNER_PROCESS = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(_PROJECT_DIR)
            )

            # Background thread to capture stdout line by line
            def log_reader(proc):
                for line in proc.stdout:
                    clean = line.strip()
                    if clean:
                        with RUNNER_LOCK:
                            RUNNER_LOGS.append(clean)
                proc.stdout.close()

            threading.Thread(target=log_reader, args=(RUNNER_PROCESS,), daemon=True).start()

    def stop_runner(self):
        global RUNNER_PROCESS, RUNNER_LOGS
        with RUNNER_LOCK:
            if RUNNER_PROCESS is not None and RUNNER_PROCESS.poll() is None:
                RUNNER_PROCESS.terminate()
                RUNNER_PROCESS = None
                RUNNER_LOGS.append("--- Process stopped by user ---")

    def run_merge(self):
        out_d = detect_output_dir()
        test_d = detect_test_dir()
        py_exe = sys.executable
        cmd = [
            py_exe,
            str(_PROJECT_DIR / "merge_shards.py"),
            "--output-dir", str(out_d),
            "--test-dir", str(test_d),
            "--num-shards", "3"
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_DIR))
            if p.returncode == 0:
                return {"status": "ok", "message": "Merged successfully! Submission files are ready."}
            else:
                return {"status": "error", "message": f"Merge failed: {p.stderr or p.stdout}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def log_message(self, format, *args):
        pass  # suppress standard HTTP request logs to keep terminal clean


def run_dashboard():
    # Find free port if 5000 is occupied
    port = PORT
    for candidate in [5000, 5050, 8080, 8888, 3000]:
        try:
            server = socketserver.TCPServer(("", candidate), DashboardHandler)
            port = candidate
            break
        except OSError:
            continue

    url = f"http://localhost:{port}"
    print("=" * 65)
    print(" BUSINESS ENTITY RESOLUTION - INTERACTIVE HUB")
    print("=" * 65)
    print(f" Dashboard URL: {url}")
    print(" Opening browser automatically...")
    print(" Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65)

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        server.server_close()


if __name__ == "__main__":
    run_dashboard()
