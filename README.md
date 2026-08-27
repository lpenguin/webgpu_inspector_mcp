# WebGPU Inspector MCP

An OpenCode / Model Context Protocol (MCP) server for [WebGPU Inspector](https://github.com/brendan-duncan/webgpu_inspector).

It allows AI assistants to drive Chrome/Chromium to capture live WebGPU frames, profile frame performance, inspect WGSL shaders, buffers, textures, diagnose WebGPU rendering issues, and step-debug WGSL compute, vertex, and fragment shaders interactively.

## Features

- **Automated Browser Instrumentation**: Launches Chromium/Chrome/Edge via CDP and injects WebGPU Inspector into every page before page scripts run.
- **Interactive Shader Debugging (8 Tools)**: Step-by-step WGSL debugging for compute, vertex, and fragment shaders with breakpoints, variable inspection, callstack unwinding, and expression evaluation.
- **Capture File Analysis & Live Inspection (21 Tools)**: Tools for browser control, frame capture, GPU timestamp profiling, live buffer/texture reads, draw state diffing, and shader inspection. Load and inspect `.wgpuc` binary files or `.json` recordings without running a browser.

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

## Shader Debugging Tools

The server provides 8 dedicated tools for interactive shader debugging powered by `wgsl_reflect`:

| Tool | Description | Key Parameters |
| --- | --- | --- |
| `shader_debug_start` | Starts an interactive debugging session for a compute dispatch, vertex draw, or fragment pixel. | `captureId`, `commandIndex`, `stage` (`"compute"`, `"vertex"`, `"fragment"`), `entryPoint`, `code`, `invocation`, `breakpoints`, `constants`, `sessionId` |
| `shader_debug_step` | Advances shader execution by stepping lines or instructions. | `sessionId`, `action` (`"step_next"`, `"step_into"`, `"step_over"`, `"step_out"`), `count` |
| `shader_debug_continue` | Continues execution until a line breakpoint is hit, the shader finishes, or `maxSteps` is reached. | `sessionId`, `maxSteps` (default: 50000) |
| `shader_debug_set_breakpoints` | Sets, removes, or clears line breakpoints in the active session. | `sessionId`, `add` (array of line numbers), `remove` (array of line numbers), `clearAll` (boolean) |
| `shader_debug_get_stack` | Returns the current callstack / call frames (innermost frame first) with function names and line numbers. | `sessionId` |
| `shader_debug_get_variables` | Inspects variables categorized by scope (`locals`, `inputs`, `globals`, `constants`, `resources`). | `sessionId`, `scope` (`"all"`, `"locals"`, `"inputs"`, `"globals"`, `"constants"`, `"resources"`), `filter`, `maxDepth` |
| `shader_debug_eval` | Evaluates path expressions (e.g. `in.position.x`, `uFrame.cameraPos.xyz`, `params.multiplier`, `matrix[1][2]`). | `sessionId`, `expression` (or `path`) |
| `shader_debug_stop` | Stops and disposes an active debug session, freeing associated resources. | `sessionId` |

### Invocation Coordinates by Stage

When starting a debug session with `shader_debug_start`, provide the target invocation:

- **Compute**: `{ "threadId": [x, y, z] }` or `{ "dispatchId": [x, y, z] }`
- **Vertex**: `{ "vertexIndex": 0, "instanceIndex": 0 }` (fetches vertex buffer attributes automatically)
- **Fragment**: `{ "pixelX": 100, "pixelY": 100 }` (interpolates vertex outputs / varyings and `@builtin(position)` at pixel center)

## Base Inspection & Profiling Tools (21 Tools)

| Tool | Description |
| --- | --- |
| `launch_browser` | Launch a new instrumented Chrome/Edge instance via CDP. |
| `attach_browser` | Attach to an already-running Chrome/Edge on a remote debugging port. |
| `open_page` | Open a new instrumented tab in the controlled browser. |
| `browser_status` | Check browser connection and instrumented pages status. |
| `list_pages` | List connected WebGPU pages. |
| `screenshot_page` | Capture a composited PNG screenshot of the presented canvas/page. |
| `capture_frames` | Request frame capture from connected page with optional GPU profiling. |
| `list_captures` | List captures currently loaded in the store. |
| `load_capture_file` | Load a `.wgpuc` or `.json` capture file from disk. |
| `get_capture_summary` | Get high-level summary of passes, draw calls, objects, and validation errors. |
| `analyze_performance` | Profile GPU duration per pass and diagnose bottlenecks (fillrate vs ALU). |
| `get_commands` | Return paginated list of GPU commands in capture. |
| `get_object` | Inspect GPU object descriptor (buffers, textures, pipelines). |
| `get_shader` | Retrieve WGSL source code of a ShaderModule. |
| `get_validation_errors` | Get WebGPU validation errors recorded during capture. |
| `get_draw_state` | Resolve pipeline, bound vertex buffers, index buffers, and bind groups for a draw/dispatch. |
| `decode_vertex_buffer` | Decode captured vertex buffer payloads into typed attribute values. |
| `diff_draws` | Structurally diff the state between two draw calls. |
| `read_buffer` | Read live GPU buffer contents from connected page. |
| `read_texture` | Read live GPU texture / render target region and return statistics and ASCII view. |
| `get_frame_stats` | Sample live frame rate, frame time variance, and CPU submit cost. |

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

