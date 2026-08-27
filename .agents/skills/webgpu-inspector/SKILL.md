---
name: webgpu-inspector
description: Capture, inspect, profile, and debug WebGPU frames, pipelines, shaders, buffers, and textures from live web pages or .wgpuc capture files using WebGPU Inspector MCP tools.
---

# WebGPU Inspector Skill

Drive Chrome/Chromium to capture and inspect WebGPU applications, profile frame performance, analyze pipelines and WGSL shaders, and inspect GPU buffers and textures using the WebGPU Inspector MCP server.

## Overview

The WebGPU Inspector MCP server connects OpenCode to running WebGPU applications in a browser via Chrome DevTools Protocol (CDP) and a localhost bridge (`ws://127.0.0.1:9690/page`). It automatically injects the WebGPU Inspector script before any page scripts run.

### Capabilities
- **Browser Automation**: Launch Chromium/Chrome/Edge or attach to an existing session with `--remote-debugging-port=9222`.
- **Live Frame Captures**: Capture single or multiple frames with GPU timestamps (`profilePasses: true`) and state inspection.
- **Performance Profiling**: Analyze frame budget, CPU submit overhead vs GPU time, pass bottlenecks (fillrate/ROP vs fragment ALU), and get optimization suggestions.
- **Live State Inspection**: Sample live FPS/dropped frames (`get_frame_stats`), read live buffer bytes (`read_buffer`), read texture/render-target regions (`read_texture`), or take full canvas screenshots (`screenshot_page`).
- **Capture File Analysis**: Load and inspect `.wgpuc` binary files or legacy `.json` capture recordings without a browser.
- **Interactive Shader Debugging**: Step through WGSL shader execution, set breakpoints, inspect call stack, view scoped variables, and evaluate expressions on captured draw/dispatch calls.

---

## MCP Tools Reference

| Tool | Description | Key Parameters |
| --- | --- | --- |
| `launch_browser` | Launches Chrome/Chromium/Edge with automated WebGPU injection | `url`, `headless`, `executablePath` |
| `attach_browser` | Attaches to a running browser with debugging enabled | `browserURL` (default: `http://localhost:9222`) |
| `open_page` | Opens a new instrumented tab and navigates to the URL | `url` |
| `browser_status` | Returns connection mode and list of instrumented pages | *(none)* |
| `list_pages` | Lists active pages connected to the bridge | *(none)* |
| `screenshot_page` | Captures a PNG screenshot of the composited WebGPU canvas/page | `instanceId`, `selector`, `fullPage` |
| `get_frame_stats` | Samples live FPS, frame times, dropped frames, CPU submit time | `pageId`, `durationMs` (100–10000) |
| `capture_frames` | Captures N WebGPU frames from a connected page | `pageId`, `count`, `profilePasses`, `payloads` |
| `list_captures` | Lists all in-memory and saved captures | *(none)* |
| `load_capture_file` | Loads a `.wgpuc` or `.json` capture from disk | `filePath` |
| `get_capture_summary` | Summary of commands, draws, passes, objects, validation errors, heuristic issues | `captureId` |
| `analyze_performance` | In-depth performance analysis: budget, CPU/GPU bound verdict, pass ranking, bottlenecks | `captureId` |
| `get_commands` | Filterable list of WebGPU commands | `captureId`, `offset`, `limit`, `method`, `passLabel` |
| `get_object` | Inspect descriptor and metadata of any GPU object (Buffer, Texture, Pipeline, etc.) | `id`, `captureId` |
| `get_shader` | Retrieves WGSL shader source code | `id`, `captureId` |
| `get_draw_state` | Resolves bound pipeline, vertex layout, bind groups, and buffers for a draw call | `commandIndex`, `captureId` |
| `decode_vertex_buffer` | Decodes first N vertices into structured attributes | `commandIndex`, `count`, `offset` |
| `diff_draws` | Diffs bound state between two draw calls | `cmdA`, `cmdB`, `captureId` |
| `get_validation_errors` | Returns all WebGPU validation errors recorded during capture | `captureId` |
| `read_buffer` | Reads and decodes live GPU buffer contents without full capture | `bufferId`, `type` (`float32`, `uint32`, `hex`, etc.), `offset`, `size` |
| `read_texture` | Reads live GPU texture/render-target region (min/max/mean, hole texels, ASCII view) | `textureId`, `mipLevel`, `x`, `y`, `width`, `height` |
| `shader_debug_start` | Starts an interactive WGSL shader debugging session (compute/vertex/fragment) | `commandIndex`, `stage`, `invocation`, `captureId`, `breakpoints`, `code` |
| `shader_debug_step` | Steps through shader execution (`step_next`, `step_over`, `step_into`, `step_out`) | `sessionId`, `action`, `count` |
| `shader_debug_continue` | Resumes execution until breakpoint or shader completion | `sessionId`, `maxSteps` |
| `shader_debug_set_breakpoints` | Sets, removes, or clears line breakpoints | `sessionId`, `add`, `remove`, `clearAll` |
| `shader_debug_get_stack` | Inspects callstack frames across nested function calls | `sessionId` |
| `shader_debug_get_variables` | Inspects scoped variables (`locals`, `inputs`, `globals`, `constants`, `resources`) | `sessionId`, `scope`, `filter`, `maxDepth` |
| `shader_debug_eval` | Evaluates variable paths, struct fields, vector swizzles (`mesh.normal.rgb`) | `sessionId`, `expression` |
| `shader_debug_stop` | Terminates and disposes active shader debug session | `sessionId` |

---

## Standard Workflows

### 1. Live Page Capture & Performance Profiling

When analyzing live WebGPU app performance (low FPS, jank, bottlenecks):

1. **Check Status / Launch Browser:**
   - Call `browser_status`. If not connected, call `launch_browser({ url: "<URL>" })`.
2. **Sample Live Frame Health (Optional Quick Check):**
   - Call `get_frame_stats({ durationMs: 1000 })` to check live FPS and whether frames are CPU-bound or GPU/vsync-bound.
3. **Capture with GPU Timing:**
   - Always pass `profilePasses: true` and `payloads: "none"` for performance analysis:
     ```json
     capture_frames({ "count": 1, "profilePasses": true, "payloads": "none" })
     ```
4. **Run Performance Analysis:**
   - Call `analyze_performance({ captureId: "<id>" })`.
   - Inspect frame-budget verdict, pass duration ranking, and fillrate vs fragment-ALU warnings.
5. **Inspect Top Hotspots:**
   - Use `get_shader` for shaders attached to the heaviest passes.
   - Use `get_commands` filtered by pass label to see draw call volume and state changes.

### 2. Debugging Rendering Errors & Correctness

When geometry is missing, rendering is corrupted, or errors are thrown:

1. **Capture Frame with Payloads:**
   - Call `capture_frames({ count: 1 })`.
2. **Check Summary & Validation Errors:**
   - Call `get_capture_summary()` and `get_validation_errors()`.
   - Note any reported WebGPU validation errors (expired canvas context, destroyed buffers/textures, bind group mismatches).
3. **Inspect Draw Calls & Vertex Data:**
   - Identify the draw command index via `get_commands({ method: "draw" })`.
   - Call `get_draw_state({ commandIndex: <index> })` to inspect active bindings and vertex buffer commands.
   - Call `decode_vertex_buffer({ commandIndex: <bufferDataCommandIndex>, count: 10 })` to verify vertex attribute values.
4. **Compare Draws:**
   - If one draw works and another fails, run `diff_draws({ cmdA: <index1>, cmdB: <index2> })`.

### 3. Analyzing Saved Capture Files (`.wgpuc`)

1. Call `load_capture_file({ filePath: "/path/to/capture.wgpuc" })`.
2. Call `get_capture_summary({ captureId: "<id>" })`.
3. If GPU timestamp queries were recorded, run `analyze_performance({ captureId: "<id>" })`.
4. Inspect WGSL with `get_shader` and object descriptors with `get_object`.

### 4. Interactive WGSL Shader Debugging

1. **Start Debug Session**:
   - For compute dispatches: `shader_debug_start({ commandIndex: <dispatchIdx>, stage: "compute", invocation: { threadId: [0, 0, 0] } })`.
   - For vertex draws: `shader_debug_start({ commandIndex: <drawIdx>, stage: "vertex", invocation: { vertexIndex: 0, instanceIndex: 0 } })`.
   - For fragment draws: `shader_debug_start({ commandIndex: <drawIdx>, stage: "fragment", invocation: { x: 400, y: 300 } })`.
2. **Step Through Code**:
   - Use `shader_debug_step({ sessionId: "<id>", action: "step_over" })` (or `step_into`, `step_out`, `step_next`).
3. **Breakpoints & Continuous Run**:
   - Add breakpoints: `shader_debug_set_breakpoints({ sessionId: "<id>", add: [42, 58] })`.
   - Run to next breakpoint: `shader_debug_continue({ sessionId: "<id>" })`.
4. **Inspect State & Evaluate Expressions**:
   - Call stack: `shader_debug_get_stack({ sessionId: "<id>" })`.
   - Scoped variables: `shader_debug_get_variables({ sessionId: "<id>", scope: "all" })`.
   - Evaluate expressions: `shader_debug_eval({ sessionId: "<id>", expression: "material.albedo.rgb" })`.
5. **Stop Session**:
   - `shader_debug_stop({ sessionId: "<id>" })`.

---

## Best Practices

- **Performance queries imply `profilePasses: true`**: Always enable timestamp queries and set `payloads: "none"` when diagnosing performance.
- **Validation errors first**: WebGPU validation errors are confirmed bugs; address them before heuristic performance tweaks.
- **Avoid unbounded queries**: Captures can contain tens of thousands of commands. Always paginate with `offset` / `limit` or filter by `method` / `passLabel`.
- **Live readbacks**: Use `read_buffer` or `read_texture` for quick live checks on specific resources without capturing entire multi-frame sequences.
