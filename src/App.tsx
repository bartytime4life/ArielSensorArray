// src/App.tsx
// -------------------------------------------------------------------------------------------------
// SpectraMind V50 — React Stub Dashboard (GUI-optional, CLI-first)
// A thin React front-end that calls the FastAPI backend contract:
//   • POST /api/run        -> run a spectramind CLI command
//   • POST /api/artifacts  -> list artifact files by glob
//   • GET  /api/log?n=...  -> tail of logs/v50_debug_log.md
//
// This file is intentionally self-contained and heavily commented for clarity. It uses Material-UI.
// Assumptions:
//   • The FastAPI backend is running on the same origin (e.g., http://localhost:8000).
//   • Static files for artifacts may be optionally served under /static (see backend notes).
//   • The React app is built with TypeScript and MUI already installed.
//     npm i @mui/material @emotion/react @emotion/styled @mui/icons-material
//
// Notes on the SpectraMind philosophy baked in here:
//   • GUI is a thin shell around CLI: no hidden state; everything flows from CLI → artifacts → logs.
//   • Auditability: every /api/run call is logged server-side into v50_debug_log.md.
//   • Reproducibility: arguments are passed exactly as Hydra/CLI would receive them.
//
// -------------------------------------------------------------------------------------------------

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Container,
  CssBaseline,
  Divider,
  FormControlLabel,
  Grid,
  IconButton,
  InputAdornment,
  Link,
  Paper,
  Snackbar,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Toolbar,
  Tooltip,
  Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import DescriptionIcon from "@mui/icons-material/Description";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";

// ----------------------------
// Types for backend contracts:
// ----------------------------
type RunRequest = {
  cli: string;
  args: string[];
  cwd?: string;
};

type RunResponse = {
  returncode: number;
  stdout: string;
  stderr: string;
  command: string;
  cwd: string;
  timestamp: string;
};

type ArtifactsRequest = {
  glob: string;
  sort?: boolean;
  limit?: number;
};

type ArtifactsResponse = {
  files: string[];
};

// ----------------------------
// Utility: simple shell-like arg parser (handles "quoted strings" and escapes)
// This keeps extra CLI args faithful when users provide quoted values.
// ----------------------------
function parseShellArgs(input: string): string[] {
  // Minimal, pragmatic parser:
  //  - Supports double quotes "..."
  //  - Supports backslash escaping within double quotes
  //  - Splits on whitespace outside quotes
  // This is not a full POSIX shell parser but sufficient for dashboard input.
  const args: string[] = [];
  let i = 0;
  const n = input.length;

  while (i < n) {
    // Skip whitespace
    while (i < n && /\s/.test(input[i])) i++;
    if (i >= n) break;

    let token = "";
    if (input[i] === '"') {
      // Quoted
      i++;
      while (i < n) {
        const ch = input[i];
        if (ch === '\\') {
          if (i + 1 < n) {
            token += input[i + 1];
            i += 2;
          } else {
            i++;
          }
        } else if (ch === '"') {
          i++;
          break;
        } else {
          token += ch;
          i++;
        }
      }
    } else {
      // Unquoted
      while (i < n && !/\s/.test(input[i])) {
        token += input[i++];
      }
    }
    if (token.length > 0) args.push(token);
  }

  return args;
}

// ----------------------------
// Main Component
// ----------------------------
const App: React.FC = () => {
  // ---------- Controls & state ----------
  const [repoRoot, setRepoRoot] = useState<string>(window.location.pathname.includes("/static") ? "/" : "");
  const [outputsDir, setOutputsDir] = useState<string>("outputs/diag_vX");
  const [cliExe, setCliExe] = useState<string>("spectramind");
  const [includeUMAP, setIncludeUMAP] = useState<boolean>(true);
  const [includeTSNE, setIncludeTSNE] = useState<boolean>(true);
  const [extraArgsInput, setExtraArgsInput] = useState<string>("");
  const [reportPath, setReportPath] = useState<string>("");
  const [reportHtmlEmbedMode, setReportHtmlEmbedMode] = useState<"static" | "inline">("static"); // 'static': iframe src, 'inline': iframe srcDoc
  const [logTail, setLogTail] = useState<string>("");
  const [stdoutText, setStdoutText] = useState<string>("");
  const [stderrText, setStderrText] = useState<string>("");
  const [lastCommand, setLastCommand] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<number>(0); // 0: Report, 1: Summary, 2: Plots, 3: Log, 4: Artifacts, 5: Console
  const [snack, setSnack] = useState<{ open: boolean; msg: string }>({ open: false, msg: "" });
  const [pollIntervalMs, setPollIntervalMs] = useState<number>(1500);
  const pollTimerRef = useRef<number | null>(null);

  // ---------- Derived CLI args ----------
  const argsArray = useMemo(() => {
    const base = ["diagnose", "dashboard", "--outputs.dir", outputsDir];
    if (!includeUMAP) base.push("--no-umap");
    if (!includeTSNE) base.push("--no-tsne");
    const extra = parseShellArgs(extraArgsInput);
    return base.concat(extra);
  }, [outputsDir, includeUMAP, includeTSNE, extraArgsInput]);

  // ---------- Helpers ----------
  const showSnack = (msg: string) => setSnack({ open: true, msg });

  const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
    const res = await fetch(url, init);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
    }
    return (await res.json()) as T;
  };

  const fetchText = async (url: string): Promise<string> => {
    const res = await fetch(url);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
    }
    return await res.text();
  };

  const updateArtifacts = useCallback(async () => {
    // Find newest diagnostics report
    const body: ArtifactsRequest = {
      glob: `${outputsDir}/**/*.html`,
      sort: true,
      limit: 20,
    };
    const data = await fetchJson<ArtifactsResponse>("/api/artifacts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (data.files?.length) {
      setReportPath(data.files[0]);
    }
    return data.files ?? [];
  }, [outputsDir]);

  const refreshLog = useCallback(async () => {
    const text = await fetchText(`/api/log?n=50000`);
    setLogTail(text);
  }, []);

  // ---------- Run flow ----------
  const onRun = useCallback(async () => {
    setRunning(true);
    setStdoutText("");
    setStderrText("");
    try {
      // Kick off CLI
      const payload: RunRequest = {
        cli: cliExe || "spectramind",
        args: argsArray,
        cwd: repoRoot || undefined,
      };
      setLastCommand(`${payload.cli} ${payload.args.join(" ")}`);
      const runRes = await fetchJson<RunResponse>("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setStdoutText(runRes.stdout || "");
      setStderrText(runRes.stderr || "");
      // Refresh artifacts and logs
      await updateArtifacts();
      await refreshLog();
      showSnack(runRes.returncode === 0 ? "CLI completed successfully" : `CLI finished with code ${runRes.returncode}`);
    } catch (err: any) {
      setStderrText(String(err?.message || err));
      showSnack(`Error: ${String(err?.message || err)}`);
    } finally {
      setRunning(false);
    }
  }, [cliExe, argsArray, repoRoot, updateArtifacts, refreshLog]);

  // ---------- Polling for log tail ----------
  useEffect(() => {
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollTimerRef.current = window.setInterval(() => {
      refreshLog().catch(() => void 0);
    }, Math.max(500, pollIntervalMs));
    return () => {
      if (pollTimerRef.current) window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    };
  }, [pollIntervalMs, refreshLog]);

  // Initial fetch log + artifacts
  useEffect(() => {
    refreshLog().catch(() => void 0);
    updateArtifacts().catch(() => void 0);
  }, [refreshLog, updateArtifacts]);

  // ---------- Render helpers ----------
  const renderReport = () => {
    if (!reportPath) {
      return (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="body2">No diagnostics HTML report found yet.</Typography>
        </Paper>
      );
    }

    // Option A: Static serving via backend (recommended when FastAPI exposes /static)
    if (reportHtmlEmbedMode === "static") {
      // If your backend mounts /static for artifact dirs, you can point iframe src to a /static path.
      // If not, the absolute file path will still work when served by the backend's StaticFiles.
      const src = reportPath.startsWith("/") ? reportPath : `/static/${reportPath}`;
      return (
        <Paper variant="outlined" sx={{ p: 0, height: "75vh", overflow: "hidden" }}>
          <iframe
            title="Diagnostics Report"
            src={src}
            style={{ border: 0, width: "100%", height: "100%" }}
          />
        </Paper>
      );
    }

    // Option B: Inline HTML (srcDoc) — requires you to fetch the file contents via an authenticated endpoint.
    // For simplicity, here we show a placeholder indicating how you'd use srcDoc.
    return (
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="body2">
          Inline mode requires fetching HTML content and passing it to <code>srcDoc</code>. For security,
          ensure the backend sanitizes HTML if needed. Consider using sandboxed iframes.
        </Typography>
      </Paper>
    );
  };

  const renderConsole = () => (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Last Command</Typography>
          <Typography variant="body2" sx={{ fontFamily: "monospace" }}>{lastCommand || "(none yet)"}</Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper variant="outlined" sx={{ p: 2, height: 300, overflow: "auto", bgcolor: "#0b0b0b" }}>
          <Typography variant="subtitle2" color="white" gutterBottom>stdout</Typography>
          <pre style={{ color: "#c7f7c7", margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{stdoutText || "(empty)"}</pre>
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper variant="outlined" sx={{ p: 2, height: 300, overflow: "auto", bgcolor: "#0b0b0b" }}>
          <Typography variant="subtitle2" color="white" gutterBottom>stderr</Typography>
          <pre style={{ color: "#ffcdd2", margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{stderrText || "(empty)"}</pre>
        </Paper>
      </Grid>
    </Grid>
  );

  // ---------- JSX ----------
  return (
    <>
      <CssBaseline />
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            SpectraMind V50 — React Stub Dashboard
          </Typography>
          <Tooltip title="Refresh Log">
            <span>
              <IconButton color="inherit" onClick={() => refreshLog().catch(() => void 0)}>
                <RefreshIcon />
              </IconButton>
            </span>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 2 }}>
        {/* Controls */}
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Repository Root (cwd)"
                value={repoRoot}
                onChange={(e) => setRepoRoot(e.target.value)}
                placeholder="/path/to/repo"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <FolderOpenIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Outputs Directory"
                value={outputsDir}
                onChange={(e) => setOutputsDir(e.target.value)}
                placeholder="outputs/diag_vX"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <DescriptionIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <TextField
                fullWidth
                label="CLI Executable"
                value={cliExe}
                onChange={(e) => setCliExe(e.target.value)}
                placeholder="spectramind"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Poll Interval (ms)"
                type="number"
                value={pollIntervalMs}
                onChange={(e) => setPollIntervalMs(Number(e.target.value) || 1500)}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={2} alignItems="center">
                <FormControlLabel
                  control={<Switch checked={includeUMAP} onChange={(e) => setIncludeUMAP(e.target.checked)} />}
                  label="Include UMAP"
                />
                <FormControlLabel
                  control={<Switch checked={includeTSNE} onChange={(e) => setIncludeTSNE(e.target.checked)} />}
                  label="Include t-SNE"
                />
                <TextField
                  fullWidth
                  label='Extra CLI Args (e.g., --flag "quoted value")'
                  value={extraArgsInput}
                  onChange={(e) => setExtraArgsInput(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <InsertDriveFileIcon />
                      </InputAdornment>
                    ),
                  }}
                />
              </Stack>
            </Grid>

            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={2} justifyContent="flex-end">
                <Button
                  variant="contained"
                  color={running ? "inherit" : "primary"}
                  startIcon={running ? <StopIcon /> : <PlayArrowIcon />}
                  onClick={onRun}
                  disabled={running}
                >
                  {running ? "Running..." : "Run Diagnose"}
                </Button>
                <Tooltip title="Toggle report embedding mode">
                  <Button
                    variant="outlined"
                    onClick={() =>
                      setReportHtmlEmbedMode((m) => (m === "static" ? "inline" : "static"))
                    }
                  >
                    Report Mode: {reportHtmlEmbedMode}
                  </Button>
                </Tooltip>
                <Tooltip title="Show current CLI args to be executed">
                  <Button variant="text" onClick={() => showSnack(argsArray.join(" "))}>
                    Preview Args
                  </Button>
                </Tooltip>
              </Stack>
            </Grid>
          </Grid>
          <Divider sx={{ my: 2 }} />
          <Typography variant="caption" color="text.secondary">
            CLI Preview:&nbsp;
            <code style={{ fontFamily: "monospace" }}>
              {cliExe || "spectramind"} {argsArray.join(" ")}
            </code>
          </Typography>
        </Paper>

        {/* Tabs */}
        <Paper variant="outlined" sx={{ mb: 2 }}>
          <Tabs
            value={activeTab}
            onChange={(_, v) => setActiveTab(v)}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Diagnostics HTML" />
            <Tab label="diagnostic_summary.json" />
            <Tab label="Plots" />
            <Tab label="Log" />
            <Tab label="Artifacts" />
            <Tab label="Console" />
          </Tabs>
        </Paper>

        {/* Panels */}
        {activeTab === 0 && (
          <Box>{renderReport()}</Box>
        )}

        {activeTab === 1 && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="body2">
              In a full implementation, this panel would locate and load
              <code> diagnostic_summary.json</code>, render per-planet tables and global metrics,
              and provide a raw JSON expander with download. This stub keeps the focus on the API contract.
            </Typography>
          </Paper>
        )}

        {activeTab === 2 && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="body2">
              Plots panel: list <code>*.png/*.jpg</code> artifacts under <code>{outputsDir}</code>,
              show them in a responsive grid with download buttons. Use <code>/api/artifacts</code> with
              glob patterns like <code>{`${outputsDir}/**/*.{png,jpg,jpeg}`}</code>.
            </Typography>
          </Paper>
        )}

        {activeTab === 3 && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={2} alignItems="center" mb={1}>
              <Typography variant="subtitle2">logs/v50_debug_log.md (tail)</Typography>
              <Tooltip title="Refresh Now">
                <IconButton size="small" onClick={() => refreshLog().catch(() => void 0)}>
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
            <Paper variant="outlined" sx={{ p: 2, height: "60vh", overflow: "auto", bgcolor: "#0b0b0b" }}>
              <pre style={{ color: "#cfd8dc", margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {logTail || "(empty)"}
              </pre>
            </Paper>
          </Paper>
        )}

        {activeTab === 4 && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="body2" paragraph>
              Artifacts browser: query files via <code>/api/artifacts</code> and render with download buttons.
              For static serving, map your artifact root to <code>/static</code> in FastAPI:
              <br />
              <code>app.mount("/static", StaticFiles(directory=&lt;artifact_root&gt;), name="static")</code>
            </Typography>
            <Button
              variant="outlined"
              onClick={() =>
                updateArtifacts()
                  .then((files) => showSnack(`${files.length} artifact(s) matched`))
                  .catch((e) => showSnack(String(e)))
              }
            >
              Refresh Artifacts
            </Button>
          </Paper>
        )}

        {activeTab === 5 && renderConsole()}

        <Box mt={3} mb={6}>
          <Typography variant="caption" color="text.secondary">
            GUI is a thin shell. All operations map to CLI + Hydra configs and are logged for audit. See{" "}
            <Link href="#" onClick={(e) => e.preventDefault()}>
              v50_debug_log.md
            </Link>{" "}
            for immutable run history.
          </Typography>
        </Box>
      </Container>

      <Snackbar
        open={snack.open}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.msg}
        autoHideDuration={3500}
      />
    </>
  );
};

export default App;
