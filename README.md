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
