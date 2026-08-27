# WebGPU Inspector MCP

An OpenCode / Model Context Protocol (MCP) server for [WebGPU Inspector](https://github.com/brendan-duncan/webgpu_inspector).

It allows AI assistants to drive Chrome/Chromium to capture live WebGPU frames, profile frame performance, inspect WGSL shaders, buffers, textures, and diagnose WebGPU rendering issues.

## Features

- **Automated Browser Instrumentation**: Launches Chromium/Chrome/Edge via CDP and injects WebGPU Inspector into every page before page scripts run.
- **21 MCP Tools**: Tools for browser control, frame capture, GPU timestamp profiling, live buffer/texture reads, draw state diffing, and shader inspection.
- **Capture File Analysis**: Load and inspect `.wgpuc` binary files or `.json` recordings without running a browser.

## Installation

```bash
git clone git@github.com:lpenguin/webgpu_inspector_mcp.git
cd webgpu_inspector_mcp
npm install
```

## MCP Configuration

Add the server to your `opencode.json` (or Claude desktop configuration):

```json
{
  "mcp": {
    "webgpu-inspector": {
      "type": "local",
      "command": ["node", "/path/to/webgpu_inspector_mcp/index.js"],
      "enabled": true,
      "environment": {
        "WEBGPU_BRIDGE_PORT": "9690",
        "WEBGPU_BRIDGE_CAPTURES_DIR": "/path/to/webgpu_inspector_mcp/captures"
      }
    }
  }
}
```

## Environment Variables

All environment variables are **optional**:

| Variable | Default | Description |
| --- | --- | --- |
| `WEBGPU_BRIDGE_CAPTURES_DIR` | `<cwd>/captures` | Directory where uploaded `.wgpuc` frame captures are stored. If not set, it defaults to creating a `captures/` folder in the current working directory. Setting an explicit absolute path keeps all captures in one dedicated location. |
| `WEBGPU_BRIDGE_PORT` | `9690` | Port for the local HTTP + WebSocket bridge that instrumented pages connect to. If the default port is already occupied by another session, the bridge automatically falls back to an OS-assigned free port. |
| `WEBGPU_BRIDGE_HOST` | `127.0.0.1` | Bind address for the local bridge server. |
| `WEBGPU_BRIDGE_CHROME` | *(auto-detected)* | Custom binary path for Google Chrome, Chromium, or Microsoft Edge. Auto-detects standard system paths if omitted. |
| `WEBGPU_BRIDGE_TOKEN` | *(none)* | Optional auth token required for instrumented pages to connect to the bridge. |
| `WEBGPU_INSPECTOR_SCRIPT` | *(bundled)* | Path or URL to custom `webgpu_inspector.js` injection script. Defaults to the bundled file. |

