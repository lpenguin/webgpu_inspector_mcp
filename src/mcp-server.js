// MCP server: the tools LLMs call to drive captures, inspect them, and debug shaders.
//
// Output discipline: tools return summaries, paginated slices, decoded windows,
// and state snapshots — never raw base64 texture/buffer blobs. Captures are multi-MB
// and would blow out the model's context.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema
} from "@modelcontextprotocol/sdk/types.js";

import {
  summarize,
  analyzePerformance,
  listCommands,
  getObject,
  getShader,
  getDrawState,
  decodeVertexBuffer,
  diffDraws
} from "webgpu_inspector/claude-plugin/server/analysis.js";

import {
  ShaderDebugSessionManager,
  formatDataValue
} from "./shader-debug-session.js";
import { prepareShaderDebugSession } from "./stage-adapters.js";

// Largest WGSL source returned by get_shader before truncation.
const MAX_SHADER_CHARS = 60000;
// Soft caps for clampResult, so no tool result blows out the model's context.
const MAX_STRING_CHARS = 20000;
const MAX_ARRAY_ITEMS = 1000;

// Defensively bound a tool result before it's serialized: long strings and big
// arrays are truncated with a clear marker rather than returned in full. This
// is a backstop — individual tools already paginate/slice — so a surprising
// large field degrades instead of flooding the context.
export function clampResult(value, depth = 0, seen = new WeakSet()) {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") {
    if (value.length > MAX_STRING_CHARS) {
      return (
        value.slice(0, MAX_STRING_CHARS) +
        `...[truncated, ${value.length - MAX_STRING_CHARS} more chars]`
      );
    }
    return value;
  }
  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "bigint"
  ) {
    return value;
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return String(value);
  }
  if (depth > 20) {
    return "[Depth limit reached]";
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value) || (typeof Buffer !== "undefined" && Buffer.isBuffer(value))) {
    return `[Binary data: ${value.byteLength || value.length} bytes]`;
  }
  if (typeof value === "object") {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);
  }
  if (Array.isArray(value)) {
    const out = value
      .slice(0, MAX_ARRAY_ITEMS)
      .map((v) => clampResult(v, depth + 1, seen));
    if (value.length > MAX_ARRAY_ITEMS) {
      out.push(`...[truncated, ${value.length - MAX_ARRAY_ITEMS} more items]`);
    }
    return out;
  }
  const out = {};
  for (const k of Object.keys(value)) {
    out[k] = clampResult(value[k], depth + 1, seen);
  }
  return out;
}

// Turn a live buffer readback ({ base64, byteLength, ... }) into typed numbers.
function decodeReadback(result, type) {
  if (!result || typeof result.base64 !== "string") {
    return result || { error: "No data returned." };
  }
  const bytes = Buffer.from(result.base64, "base64");
  const out = {
    bufferId: result.bufferId,
    offset: result.offset || 0,
    byteLength: bytes.length,
    type
  };
  if (result.truncated) {
    out.truncated = result.truncated;
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  if (type === "hex") {
    out.hex = bytes.toString("hex");
    return out;
  }
  const sizes = { uint8: 1, uint16: 2, uint32: 4, int32: 4, float32: 4 };
  const step = sizes[type] || 4;
  for (let i = 0; i + step <= bytes.length; i += step) {
    switch (type) {
      case "uint8":
        values.push(dv.getUint8(i));
        break;
      case "uint16":
        values.push(dv.getUint16(i, true));
        break;
      case "uint32":
        values.push(dv.getUint32(i, true));
        break;
      case "int32":
        values.push(dv.getInt32(i, true));
        break;
      default:
        values.push(dv.getFloat32(i, true));
        break;
    }
  }
  out.values = values;
  return out;
}

function _round(v) {
  return typeof v === "number" && isFinite(v) ? Math.round(v * 1000) / 1000 : v;
}

// IEEE-754 half (uint16) -> float.
function halfToFloat(h) {
  const s = (h & 0x8000) >> 15,
    e = (h & 0x7c00) >> 10,
    f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 0x1f) {
    return f ? NaN : s ? -Infinity : Infinity;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// Decode a readTexture payload (format, region, bytesPerRow, channels, base64) into a
// model-friendly summary: per-channel min/max/mean, the fraction of "hole" texels (RGB
// all ~0 — the G-buffer clear showing through where no triangle rasterised), and a small
// ASCII luminance view of the spatial pattern. Never echoes raw pixels.
function summarizeTexture(r) {
  if (!r) {
    return { error: "No texture data returned." };
  }
  if (r.error) {
    return { error: r.error };
  }
  if (typeof r.base64 !== "string") {
    return { error: "No pixel data returned." };
  }
  const bytes = Buffer.from(r.base64, "base64");
  const { format, width, height, bytesPerRow } = r;
  const channels = r.channels || 4;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let chBytes, decode;
  if (/16float/.test(format)) {
    chBytes = 2;
    decode = (o) => halfToFloat(dv.getUint16(o, true));
  } else if (/32float/.test(format)) {
    chBytes = 4;
    decode = (o) => dv.getFloat32(o, true);
  } else if (/(rgba8|bgra8|rg8|r8|rgb10)/.test(format)) {
    chBytes = 1;
    decode = (o) => dv.getUint8(o) / 255;
  } else {
    return { format, width, height, error: `No decoder for format "${format}" yet.` };
  }
  const texelBytes = channels * chBytes;
  const bgra = format.startsWith("bgra");
  const px = (x, y) => {
    const base = y * bytesPerRow + x * texelBytes;
    const v = [];
    for (let c = 0; c < channels; c++) {
      v.push(base + c * chBytes + chBytes <= bytes.length ? decode(base + c * chBytes) : 0);
    }
    if (bgra && channels >= 3) {
      const t = v[0];
      v[0] = v[2];
      v[2] = t;
    }
    return v;
  };
  // Per-channel stats over a sampled subset (~20k samples max).
  const stride = Math.max(1, Math.floor(Math.sqrt((width * height) / 20000)));
  const mins = new Array(channels).fill(Infinity);
  const maxs = new Array(channels).fill(-Infinity);
  const sums = new Array(channels).fill(0);
  let n = 0,
    holes = 0;
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const v = px(x, y);
      n++;
      let allZero = true;
      for (let c = 0; c < channels; c++) {
        const val = v[c];
        if (val < mins[c]) {
          mins[c] = val;
        }
        if (val > maxs[c]) {
          maxs[c] = val;
        }
        sums[c] += val;
        if (c < 3 && Math.abs(val) > 1e-4) {
          allZero = false;
        }
      }
      if (allZero) {
        holes++;
      }
    }
  }
  const channelStats = [];
  for (let c = 0; c < channels; c++) {
    channelStats.push({
      channel: c,
      min: _round(mins[c]),
      max: _round(maxs[c]),
      mean: _round(sums[c] / Math.max(1, n))
    });
  }
  // ASCII spatial view (point-sampled luminance), so the model can SEE the pattern.
  const gw = Math.min(width, 72);
  const gh = Math.max(1, Math.min(height, Math.round((gw * height) / width / 2)));
  const ramp = " .:-=+*#%@";
  const rows = [];
  for (let gy = 0; gy < gh; gy++) {
    let row = "";
    for (let gx = 0; gx < gw; gx++) {
      const x = Math.min(width - 1, Math.floor(((gx + 0.5) * width) / gw));
      const y = Math.min(height - 1, Math.floor(((gy + 0.5) * height) / gh));
      const v = px(x, y);
      let lum = channels >= 3 ? 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2] : v[0];
      if (!(lum >= 0 && lum <= 1)) {
        lum = maxs[0] > mins[0] ? (v[0] - mins[0]) / (maxs[0] - mins[0]) : 0;
      }
      lum = Math.max(0, Math.min(0.9999, lum));
      row += ramp[Math.floor(lum * ramp.length)];
    }
    rows.push(row);
  }
  return {
    format,
    region: { x: r.x, y: r.y, width, height, mipLevel: r.mipLevel, layer: r.layer },
    mipSize: { width: r.mipWidth, height: r.mipHeight },
    sampleType: r.sampleType,
    channels,
    channelStats,
    holeFraction: _round(holes / Math.max(1, n)),
    holeNote:
      "fraction of sampled texels with RGB all ~0 — for a G-buffer that's the clear value showing through (unrasterised pixels / holes).",
    asciiView: rows
  };
}

/**
 * Creates an MCP server instance with all base WebGPU Inspector tools and shader debugging tools.
 *
 * @param {Object} deps
 * @param {import("webgpu_inspector/claude-plugin/server/capture-store.js").CaptureStore} deps.store
 * @param {Object} deps.bridge
 * @param {Object} deps.browser
 * @param {ShaderDebugSessionManager} [deps.sessionManager]
 * @param {string} [deps.version]
 * @returns {Server}
 */
export function createMcpServer(deps = {}) {
  const store = deps.store;
  const bridge = deps.bridge;
  const browser = deps.browser;
  const sessionManager = deps.sessionManager || new ShaderDebugSessionManager();

  const server = new Server(
    { name: "webgpu-inspector", version: deps.version || "1.0.0" },
    { capabilities: { tools: {} } }
  );

  server.sessionManager = sessionManager;

  // Resolve a captureId argument to its JSON; falls back to the most recent
  // capture when the argument is omitted.
  function resolveCapture(args) {
    let id = args && args.captureId;
    if (!id) {
      const latest = store?.latest ? store.latest() : null;
      if (!latest) {
        throw new Error("No captures available. Use capture_frames or load_capture_file first.");
      }
      id = latest.id;
    }
    const json = store?.getJson ? store.getJson(id) : null;
    if (!json) {
      throw new Error(`No capture "${id}". Use list_captures to see what is available.`);
    }
    return { id, json };
  }

  const tools = [
    // -------------------------------------------------------------------------
    // Base 21 WebGPU Inspector tools
    // -------------------------------------------------------------------------
    {
      name: "launch_browser",
      description:
        "Launch a new Chrome/Edge instance controlled by this plugin. Every page " +
        "it opens is automatically instrumented with the WebGPU Inspector — no extension and " +
        "no page changes needed. Optionally navigate a first tab to a URL.",
      inputSchema: {
        type: "object",
        properties: {
          url: { type: "string", description: "Optional URL to open in a first instrumented tab." },
          headless: { type: "boolean", description: "Run headless (default false; WebGPU usually needs headful)." },
          executablePath: { type: "string", description: "Path to the Chrome/Edge binary (auto-detected if omitted)." }
        }
      }
    },
    {
      name: "attach_browser",
      description:
        "Attach to an already-running Chrome/Edge that was started with " +
        "--remote-debugging-port. New tabs and navigations are instrumented automatically.",
      inputSchema: {
        type: "object",
        properties: {
          browserURL: { type: "string", description: "Debugger URL (default http://localhost:9222)." },
          reloadPages: { type: "boolean", description: "Reload already-open tabs so they get instrumented now (default false)." }
        }
      }
    },
    {
      name: "open_page",
      description:
        "Open a new instrumented tab in the controlled browser and navigate it to " +
        "a URL. Waits for the page to connect to the bridge and returns it ready to capture.",
      inputSchema: {
        type: "object",
        properties: {
          url: { type: "string", description: "URL to open." }
        },
        required: ["url"]
      }
    },
    {
      name: "browser_status",
      description:
        "Report whether a controlled browser is connected, how it was connected, " +
        "which targets are instrumented, and which pages have connected to the bridge.",
      inputSchema: { type: "object", properties: {} }
    },
    {
      name: "list_pages",
      description:
        "List browser pages currently connected to the live bridge. " +
        "A page connects after it is instrumented (via launch_browser/open_page) or after " +
        "it calls webgpuInspector.initializeServer() itself.",
      inputSchema: { type: "object", properties: {} }
    },
    {
      name: "screenshot_page",
      description:
        "Capture a PNG screenshot of an instrumented page and return it as an image. " +
        "This reads the COMPOSITED page — the WebGPU canvas exactly as it was presented — so it " +
        "is the reliable way to SEE what an engine rendered, independent of how it pools/aliases " +
        "its render targets (reading a pooled target back after the frame is unreliable; the " +
        "presented surface is not). Requires a page opened via launch_browser/open_page. Use this " +
        "to visually verify a rendering change, not to inspect intermediate G-buffer contents " +
        "(use read_texture / capture_frames for those).",
      inputSchema: {
        type: "object",
        properties: {
          pageId: { type: "string", description: "Page to screenshot. Optional when exactly one page is connected." },
          selector: { type: "string", description: "Optional CSS selector to clip the shot to one element (e.g. \"canvas\")." },
          fullPage: { type: "boolean", description: "Capture the full scrollable page instead of just the viewport (default false). Ignored when selector is set." }
        }
      }
    },
    {
      name: "capture_frames",
      description:
        "Ask a connected page to capture one or more WebGPU frames, then " +
        "return a summary of the resulting capture (command/draw/pass counts, object " +
        "counts, validation errors, flagged issues). Use list_pages first if unsure. " +
        "For a PERFORMANCE analysis, set profilePasses:true (and payloads:\"none\" for a " +
        "light perf-only capture) to measure per-pass GPU time, then call analyze_performance " +
        "— treat a performance request as implying profilePasses:true.",
      inputSchema: {
        type: "object",
        properties: {
          pageId: {
            type: "string",
            description: "Page to capture from. Optional when exactly one page is connected."
          },
          frames: {
            type: "integer",
            description: "Number of frames to capture (default 1).",
            minimum: 1
          },
          maxBufferSize: {
            type: "integer",
            description:
              "Optional cap, in bytes, applied to EVERY captured buffer payload " +
              "(vertex/index/storage/uniform/indirect). Buffers larger than this are truncated " +
              "to the first N bytes (recorded as truncated). Default 64KB. Use -1 to disable."
          },
          maxTextureSize: {
            type: "integer",
            description:
              "Optional cap, in bytes, on each captured texture's pixel data. Textures " +
              "larger than this are skipped (descriptor still recorded), keeping captures light. " +
              "Default 16MB. Use -1 to capture all texture data."
          },
          passLabel: {
            type: "string",
            description:
              "Optional: only capture heavy payloads (buffers/textures) for render/" +
              "compute passes whose label matches this regular expression. Greatly shrinks " +
              "captures of large frames by skipping unrelated passes (shadows, IBL, post)."
          },
          passType: {
            type: "string",
            enum: ["render", "compute"],
            description: "Optional: only capture heavy payloads for passes of this type."
          },
          payloads: {
            type: "string",
            enum: ["all", "none", "buffers", "textures"],
            description:
              "Convenience over maxBufferSize/maxTextureSize for capturing a large frame " +
              "(e.g. a full game world) without a heavy upload. \"none\" records only the command " +
              "list, object graph and validation errors — no buffer/texture bytes — which is the " +
              "fastest way to inspect draw calls, passes and pipeline usage at scale. \"buffers\" or " +
              "\"textures\" keeps only that payload kind. \"all\" (default) keeps both, subject to the " +
              "maxBufferSize/maxTextureSize caps. Explicit maxBufferSize/maxTextureSize override this."
          },
          profilePasses: {
            type: "boolean",
            description:
              "Inject per-pass GPU timestamp queries so each render/compute pass in the " +
              "capture carries a measured GPU duration (surfaced in get_capture_summary's passes " +
              "breakdown as durationMs, and as frameGpuTimeMs total). Requires the adapter to support " +
              "the \"timestamp-query\" feature (the inspector enables it automatically when available); " +
              "a no-op otherwise. Default false. Pairs well with payloads:\"none\" for a light perf pass."
          }
        }
      }
    },
    {
      name: "list_captures",
      description:
        "List captures currently available to analyze (both live captures " +
        "and capture files that were explicitly loaded).",
      inputSchema: { type: "object", properties: {} }
    },
    {
      name: "load_capture_file",
      description:
        "Load a WebGPU Inspector capture file from disk (a .wgpuc binary saved by " +
        "saveCaptureData() or the DevTools 'Save Capture' action, or a legacy .json capture) " +
        "so it can be analyzed.",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the capture file (.wgpuc, or legacy .json)." }
        },
        required: ["path"]
      }
    },
    {
      name: "get_capture_summary",
      description:
        "Summarize a capture: object counts by type, command counts by method, " +
        "derived render statistics, shader entry points, validation error count, and " +
        "heuristic performance/correctness issues. When the capture was taken with " +
        "profilePasses:true it also includes gpuTiming (per-frame GPU time, slowest pass); " +
        "when taken from a live page it includes frameBudget with a CPU/GPU/vsync bound " +
        "verdict. For a focused performance report, use analyze_performance instead.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          includeMethodCounts: { type: "boolean", description: "Include the per-method command counts map (default true)." },
          includePasses: { type: "boolean", description: "Include the per-pass breakdown (label, command range, draw/dispatch/bind counts) grouped by render/compute pass (default true)." },
          includeIssues: { type: "boolean", description: "Include heuristic performance/correctness issues (default true)." }
        }
      }
    },
    {
      name: "analyze_performance",
      description:
        "Diagnose a capture's performance and return concrete improvement suggestions. " +
        "Reports the frame budget and a CPU- / GPU- / vsync-bound verdict (needs a live capture for " +
        "CPU/refresh context and profilePasses:true for GPU time), render passes ranked by GPU time " +
        "(or fill workload when untimed) each annotated with render-target size/format/MSAA, blend " +
        "usage, fragment-shader complexity, and a likely bottleneck (fillrate/ROP vs fragment-ALU), " +
        "plus heuristic issues and ranked suggestions. Best paired with a capture taken via " +
        "capture_frames({ profilePasses: true, payloads: \"none\" }).",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." }
        }
      }
    },
    {
      name: "get_commands",
      description:
        "Return a paginated, base64-stripped slice of a capture's command list. " +
        "Each entry has its index, method, pass number, object, and arguments.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          offset: { type: "integer", description: "Start index into the (filtered) list.", minimum: 0 },
          limit: { type: "integer", description: "Max entries to return (default 50, max 500).", minimum: 1 },
          method: { type: "string", description: "Optional: only commands with this method name." },
          passLabel: { type: "string", description: "Optional regex: only commands inside a render/compute pass whose label matches." }
        }
      }
    },
    {
      name: "get_object",
      description:
        "Return one GPU object record from a capture (descriptor, label, " +
        "stacktrace), with base64 payloads omitted.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          objectId: { type: "integer", description: "Numeric object id." }
        },
        required: ["objectId"]
      }
    },
    {
      name: "get_shader",
      description: "Return the WGSL source code of a ShaderModule object in a capture.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          objectId: { type: "integer", description: "Numeric id of the ShaderModule object." }
        },
        required: ["objectId"]
      }
    },
    {
      name: "get_validation_errors",
      description: "Return the WebGPU validation errors recorded during a capture.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." }
        }
      }
    },
    {
      name: "get_draw_state",
      description:
        "Resolve the full GPU state for a draw/dispatch command: the bound pipeline " +
        "(and its vertex layout), bind groups per slot (with resource ids), vertex buffers per " +
        "slot (each with the command index that captured its bytes), the index buffer, and draw " +
        "params. Use this to diagnose what a specific draw actually read.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          commandIndex: { type: "integer", description: "Index of a draw*/dispatch* command (see get_commands).", minimum: 0 }
        },
        required: ["commandIndex"]
      }
    },
    {
      name: "decode_vertex_buffer",
      description:
        "Decode the first N vertices of a captured vertex buffer into per-attribute " +
        "numbers, so you can read e.g. 'attribute @location(2) (uv) = (0,0)' directly. Pass the " +
        "bufferDataCommandIndex from get_draw_state (a setVertexBuffer command); the vertex layout " +
        "is taken from the draw's pipeline automatically (or pass `layout`/`pipelineId`).",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          commandIndex: { type: "integer", description: "Index of the setVertexBuffer command (vertexBuffers[].bufferDataCommandIndex from get_draw_state).", minimum: 0 },
          firstN: { type: "integer", description: "Number of vertices to decode (default 8).", minimum: 1 },
          baseVertex: { type: "integer", description: "First vertex to decode from (default 0)." },
          pipelineId: { type: "integer", description: "Optional: pipeline object id to derive the layout from." },
          layout: {
            type: "object",
            description: "Optional explicit GPUVertexBufferLayout { arrayStride, attributes:[{shaderLocation, format, offset}] }."
          }
        },
        required: ["commandIndex"]
      }
    },
    {
      name: "diff_draws",
      description:
        "Structurally diff the resolved state (pipeline, bind groups, vertex/index " +
        "bindings, draw params) of two draw commands — useful when a working draw and a broken " +
        "draw share a pipeline and the difference must be in bound resources.",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          cmdA: { type: "integer", description: "Command index of the first draw.", minimum: 0 },
          cmdB: { type: "integer", description: "Command index of the second draw.", minimum: 0 }
        },
        required: ["cmdA", "cmdB"]
      }
    },
    {
      name: "read_buffer",
      description:
        "Read the current contents of a live GPU buffer on a connected page, without " +
        "taking a full capture. The inspector copies the buffer to a readback buffer, maps it, and " +
        "returns the bytes decoded as the requested type. The source buffer must have been created " +
        "with COPY_SRC usage (buffers created while a capture is armed are given COPY_SRC).",
      inputSchema: {
        type: "object",
        properties: {
          pageId: { type: "string", description: "Page to read from. Optional when exactly one page is connected." },
          bufferId: { type: "integer", description: "Numeric id of the GPU Buffer object to read." },
          offset: { type: "integer", description: "Byte offset to start reading at (default 0).", minimum: 0 },
          size: { type: "integer", description: "Number of bytes to read (default: to end of buffer, capped).", minimum: 1 },
          type: {
            type: "string",
            enum: ["uint8", "uint16", "uint32", "int32", "float32", "hex"],
            description: "How to decode the returned bytes (default float32)."
          }
        },
        required: ["bufferId"]
      }
    },
    {
      name: "read_texture",
      description:
        "Read a region of a live GPU texture / render target (G-buffer attachment, depth, " +
        "canvas) on a connected page, without taking a full capture. Copies the texture to a readback " +
        "buffer, decodes per its format, and returns per-channel min/max/mean, the fraction of " +
        "unrasterised 'hole' texels (RGB all ~0 = the clear value showing through), and a small ASCII " +
        "luminance view of the spatial pattern — never raw pixels. Pass a Texture id OR a TextureView " +
        "id (a render pass attachment is a TextureView — it is resolved to its source texture " +
        "automatically, so no get_object round-trip is needed). The texture must have COPY_SRC, which " +
        "the inspector adds to every texture while a capture is armed. Note: reading an engine's POOLED " +
        "render target after the frame can be unreliable (the pool may have recycled it); to see the " +
        "final image use screenshot_page instead.",
      inputSchema: {
        type: "object",
        properties: {
          pageId: { type: "string", description: "Page to read from. Optional when exactly one page is connected." },
          textureId: { type: "integer", description: "Numeric id of the GPU Texture OR TextureView object to read (a view is resolved to its texture)." },
          mipLevel: { type: "integer", description: "Mip level (default 0).", minimum: 0 },
          layer: { type: "integer", description: "Array layer / depth slice (default 0).", minimum: 0 },
          x: { type: "integer", description: "Region origin x (default 0).", minimum: 0 },
          y: { type: "integer", description: "Region origin y (default 0).", minimum: 0 },
          width: { type: "integer", description: "Region width (default: to the right edge, capped 1024).", minimum: 1 },
          height: { type: "integer", description: "Region height (default: to the bottom edge, capped 1024).", minimum: 1 }
        },
        required: ["textureId"]
      }
    },
    {
      name: "get_frame_stats",
      description:
        "Sample a live page's frame-health metrics over a short window (without taking a " +
        "capture) and return aggregates: frame rate (fps), average and worst frame time, dropped " +
        "frames, CPU submit time (main-thread cost per frame), the estimated display refresh interval, " +
        "and a CPU-vs-GPU/vsync bound verdict. Use this to tell whether a page is dropping frames and " +
        "whether it's CPU/main-thread bound; GPU time is not measured live, so a non-CPU verdict reads " +
        "\"GPU/vsync\" — take a capture with profilePasses:true to measure GPU time and confirm. " +
        "Requires the page's rendering loop to use requestAnimationFrame.",
      inputSchema: {
        type: "object",
        properties: {
          pageId: { type: "string", description: "Page to sample. Optional when exactly one page is connected." },
          durationMs: { type: "integer", description: "Sampling window in milliseconds (default 1000, clamped 100–10000).", minimum: 100, maximum: 10000 }
        }
      }
    },

    // -------------------------------------------------------------------------
    // 8 Shader Debugging Tools
    // -------------------------------------------------------------------------
    {
      name: "shader_debug_start",
      description:
        "Start an interactive shader debugging session for a compute dispatch, vertex draw, or fragment pixel. " +
        "Returns initial debugger state snapshot (currentLine, currentFunction, sourceSnippet, callstackDepth, activeBreakpoints, status, invocation).",
      inputSchema: {
        type: "object",
        properties: {
          captureId: { type: "string", description: "Capture id (default: most recent)." },
          commandIndex: {
            type: "integer",
            description: "Index of the draw*/dispatch* command to debug. Auto-detected if omitted.",
            minimum: 0
          },
          stage: {
            type: "string",
            enum: ["compute", "vertex", "fragment"],
            description: "Shader stage to debug (compute, vertex, fragment). Auto-inferred if omitted."
          },
          entryPoint: {
            type: "string",
            description: "Entry point function name (e.g. 'main', 'vs_main', 'fs_main'). Auto-inferred if omitted."
          },
          code: {
            type: "string",
            description: "Optional explicit WGSL source code override."
          },
          invocation: {
            type: "object",
            description:
              "Invocation coordinates / parameters depending on stage:\n" +
              "- compute: { threadId: [x,y,z] } or { dispatchId: [x,y,z] }\n" +
              "- vertex: { vertexIndex: 0, instanceIndex: 0 }\n" +
              "- fragment: { pixelX: 100, pixelY: 100, x: 100, y: 100 }"
          },
          breakpoints: {
            type: "array",
            items: { type: "integer" },
            description: "Optional line numbers to set as initial breakpoints."
          },
          constants: {
            type: "object",
            description: "Optional pipeline override constants map."
          },
          sessionId: {
            type: "string",
            description: "Optional custom session ID. Auto-generated if omitted."
          }
        }
      }
    },
    {
      name: "shader_debug_step",
      description:
        "Step execution in an active shader debugging session (step_over, step_into, step_out, step_next). " +
        "Returns updated debugger state snapshot.",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          },
          action: {
            type: "string",
            enum: ["step_over", "step_into", "step_out", "step_next"],
            description: "Stepping action (default: 'step_next')."
          },
          count: {
            type: "integer",
            description: "Number of steps to advance (default: 1).",
            minimum: 1
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_continue",
      description:
        "Continue execution in an active shader debugging session until a breakpoint is hit, the shader finishes, or maxSteps is reached.",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          },
          maxSteps: {
            type: "integer",
            description: "Maximum number of steps before pausing (default: 50000).",
            minimum: 1
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_set_breakpoints",
      description:
        "Set, remove, or clear line breakpoints in an active shader debugging session.",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          },
          add: {
            type: "array",
            items: { type: "integer" },
            description: "Line numbers to add as breakpoints."
          },
          remove: {
            type: "array",
            items: { type: "integer" },
            description: "Line numbers to remove from breakpoints."
          },
          clearAll: {
            type: "boolean",
            description: "If true, clear all existing breakpoints before applying adds."
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_get_stack",
      description:
        "Get the current callstack / call frames of an active shader debugging session (innermost frame first).",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_get_variables",
      description:
        "Inspect variables in an active shader debugging session, categorized by scope (locals, inputs, globals, constants, resources).",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          },
          scope: {
            type: "string",
            enum: ["all", "locals", "inputs", "globals", "constants", "resources"],
            description: "Scope filter (default: 'all')."
          },
          filter: {
            type: "string",
            description: "Optional substring filter to match variable names."
          },
          maxDepth: {
            type: "integer",
            description: "Maximum depth for formatting nested struct/array values (default: 2).",
            minimum: 1,
            maximum: 10
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_eval",
      description:
        "Evaluate a path expression (e.g. 'myVar', 'pos.x', 'in.uv[0]', 'material.color.rgb') in the active shader debugging session context.",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session."
          },
          expression: {
            type: "string",
            description:
              "Path expression to evaluate (e.g. 'scaledX', 'params.multiplier', 'out.color.xyz', 'buffer[0]')."
          },
          path: {
            type: "string",
            description: "Alias for expression."
          }
        },
        required: ["sessionId"]
      }
    },
    {
      name: "shader_debug_stop",
      description:
        "Stop and dispose an active shader debugging session, freeing associated resources.",
      inputSchema: {
        type: "object",
        properties: {
          sessionId: {
            type: "string",
            description: "ID of the active debugging session to stop."
          }
        },
        required: ["sessionId"]
      }
    }
  ];

  const handlers = {
    // -------------------------------------------------------------------------
    // Base tools handlers
    // -------------------------------------------------------------------------
    launch_browser: async (args) => {
      if (!browser) throw new Error("Browser controller is not configured.");
      const result = await browser.launch({
        url: args.url,
        headless: args.headless,
        executablePath: args.executablePath
      });
      const out = { browser: browser.status() };
      if (result.opened && result.opened.instanceId) {
        out.targetUrl = result.opened.url;
        out.openedPage = bridge ? await bridge.waitForPage(result.opened.instanceId, 20000) : null;
        if (!out.openedPage) {
          out.note =
            "The tab opened but did not connect to the bridge within 20s. " +
            "It may not use WebGPU yet, or it created WebGPU objects before the page finished loading.";
        }
      }
      return out;
    },

    attach_browser: async (args) => {
      if (!browser) throw new Error("Browser controller is not configured.");
      const status = await browser.attach({
        browserURL: args.browserURL,
        reloadPages: args.reloadPages
      });
      return { browser: status, pages: bridge ? bridge.listPages() : [] };
    },

    open_page: async (args) => {
      if (!args || !args.url) {
        throw new Error("url is required.");
      }
      if (!browser) throw new Error("Browser controller is not configured.");
      const opened = await browser.openPage(args.url);
      const page = bridge ? await bridge.waitForPage(opened.instanceId, 20000) : null;
      return {
        targetUrl: opened.url,
        openedPage: page,
        note: page
          ? undefined
          : "The tab opened but did not connect to the bridge within 20s. It may not " +
            "use WebGPU, or there may be an error in the page console."
      };
    },

    browser_status: async () => ({
      browser: browser ? browser.status() : { isConnected: false },
      pages: bridge ? bridge.listPages() : []
    }),

    list_pages: async () => ({
      bridgeListening: bridge ? bridge.isListening() : false,
      pages: bridge ? bridge.listPages() : []
    }),

    screenshot_page: async (args) => {
      if (!browser) throw new Error("Browser controller is not configured.");
      const instanceId = bridge ? bridge.pageInstanceId(args.pageId) : null;
      const shot = await browser.screenshot(instanceId, {
        selector: args.selector,
        fullPage: args.fullPage
      });
      const meta = {
        url: shot.url,
        mimeType: shot.mimeType,
        byteLength: shot.byteLength,
        selector: shot.selector,
        viewport: shot.viewport
      };
      return {
        __mcpContent: [
          { type: "image", data: shot.base64, mimeType: shot.mimeType },
          { type: "text", text: JSON.stringify(meta, null, 2) }
        ]
      };
    },

    capture_frames: async (args) => {
      if (!bridge) throw new Error("Bridge is not configured.");
      let maxBufferSize = args.maxBufferSize;
      let maxTextureSize = args.maxTextureSize;
      if (args.payloads && args.payloads !== "all") {
        const keepBuffers = args.payloads === "buffers";
        const keepTextures = args.payloads === "textures";
        if (maxBufferSize === undefined) {
          maxBufferSize = keepBuffers ? -1 : 0;
        }
        if (maxTextureSize === undefined) {
          maxTextureSize = keepTextures ? -1 : 0;
        }
      }
      const meta = await bridge.requestCapture({
        pageId: args.pageId,
        frames: args.frames,
        maxBufferSize,
        maxTextureSize,
        passLabel: args.passLabel,
        passType: args.passType,
        captureTimestamps: !!args.profilePasses
      });
      const json = store.getJson(meta.id);
      return { captureId: meta.id, summary: summarize(json) };
    },

    list_captures: async () => ({
      captures: store ? store.list() : []
    }),

    load_capture_file: async (args) => {
      if (!args || !args.path) {
        throw new Error("path is required.");
      }
      if (!store) throw new Error("Capture store is not configured.");
      const meta = await store.addFile(args.path);
      const json = store.getJson(meta.id);
      return { captureId: meta.id, summary: summarize(json) };
    },

    get_capture_summary: async (args) => {
      const { id, json } = resolveCapture(args);
      return {
        captureId: id,
        summary: summarize(json, {
          includeMethodCounts: args.includeMethodCounts,
          includePasses: args.includePasses,
          includeIssues: args.includeIssues
        })
      };
    },

    analyze_performance: async (args) => {
      const { id, json } = resolveCapture(args);
      return { captureId: id, performance: analyzePerformance(json) };
    },

    get_commands: async (args) => {
      const { id, json } = resolveCapture(args);
      return {
        captureId: id,
        ...listCommands(json, {
          offset: args.offset,
          limit: args.limit,
          method: args.method,
          passLabel: args.passLabel
        })
      };
    },

    get_object: async (args) => {
      const { id, json } = resolveCapture(args);
      const obj = getObject(json, args.objectId);
      if (!obj) {
        throw new Error(`No object #${args.objectId} in capture ${id}.`);
      }
      return { captureId: id, object: obj };
    },

    get_shader: async (args) => {
      const { id, json } = resolveCapture(args);
      const shader = getShader(json, args.objectId);
      if (typeof shader.code === "string" && shader.code.length > MAX_SHADER_CHARS) {
        shader.codeTruncated = {
          totalChars: shader.code.length,
          returnedChars: MAX_SHADER_CHARS
        };
        shader.code =
          shader.code.slice(0, MAX_SHADER_CHARS) +
          `\n...[truncated, ${shader.code.length - MAX_SHADER_CHARS} more chars]`;
      }
      return { captureId: id, ...shader };
    },

    get_validation_errors: async (args) => {
      const { id, json } = resolveCapture(args);
      return {
        captureId: id,
        validationErrors: Array.isArray(json.validationErrors)
          ? json.validationErrors
          : []
      };
    },

    get_draw_state: async (args) => {
      const { id, json } = resolveCapture(args);
      return {
        captureId: id,
        drawState: getDrawState(json, args.commandIndex | 0)
      };
    },

    decode_vertex_buffer: async (args) => {
      const { id, json } = resolveCapture(args);
      const resolver = (payloadId) => store.getPayload(id, payloadId);
      const decoded = decodeVertexBuffer(
        json,
        {
          commandIndex: args.commandIndex | 0,
          firstN: args.firstN,
          baseVertex: args.baseVertex,
          pipelineId: args.pipelineId,
          layout: args.layout
        },
        resolver
      );
      return { captureId: id, ...decoded };
    },

    diff_draws: async (args) => {
      const { id, json } = resolveCapture(args);
      return {
        captureId: id,
        ...diffDraws(json, args.cmdA | 0, args.cmdB | 0)
      };
    },

    read_buffer: async (args) => {
      if (!bridge) throw new Error("Bridge is not configured.");
      const result = await bridge.requestRead({
        pageId: args.pageId,
        bufferId: args.bufferId,
        offset: args.offset,
        size: args.size
      });
      return decodeReadback(result, args.type || "float32");
    },

    read_texture: async (args) => {
      if (!bridge) throw new Error("Bridge is not configured.");
      const result = await bridge.requestReadTexture({
        pageId: args.pageId,
        textureId: args.textureId,
        mipLevel: args.mipLevel,
        layer: args.layer,
        x: args.x,
        y: args.y,
        width: args.width,
        height: args.height
      });
      return summarizeTexture(result);
    },

    get_frame_stats: async (args) => {
      if (!bridge) throw new Error("Bridge is not configured.");
      return await bridge.requestFrameStats({
        pageId: args.pageId,
        durationMs: args.durationMs
      });
    },

    // -------------------------------------------------------------------------
    // 8 Shader Debugging Handlers
    // -------------------------------------------------------------------------
    shader_debug_start: async (args) => {
      const { id: captureId, json: captureJson } = resolveCapture(args);
      const payloadResolver = (payloadId) =>
        store?.getPayload ? store.getPayload(captureId, payloadId) : null;

      const session = prepareShaderDebugSession({
        capture: captureJson,
        commandIndex:
          args.commandIndex !== undefined ? Number(args.commandIndex) : undefined,
        stage: args.stage,
        entryPoint: args.entryPoint,
        code: args.code,
        invocation: args.invocation || {},
        payloadResolver,
        options: {
          constants: args.constants,
          ...(args.options || {})
        },
        sessionId: args.sessionId
      });

      if (Array.isArray(args.breakpoints) && args.breakpoints.length > 0) {
        session.setBreakpoints({ add: args.breakpoints });
      }

      sessionManager.sessions.set(session.id, session);

      return {
        captureId,
        ...session.getStateSnapshot()
      };
    },

    shader_debug_step: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      const action = args.action || "step_next";
      const count = args.count !== undefined ? Number(args.count) : 1;
      return session.step(action, count);
    },

    shader_debug_continue: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      const maxSteps =
        args.maxSteps !== undefined ? Number(args.maxSteps) : 50000;
      return session.continueExecution(maxSteps);
    },

    shader_debug_set_breakpoints: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      const activeBreakpoints = session.setBreakpoints({
        add: args.add || [],
        remove: args.remove || [],
        clearAll: !!args.clearAll
      });
      return {
        sessionId: session.id,
        activeBreakpoints
      };
    },

    shader_debug_get_stack: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      const callstack = session.getCallstack();
      return {
        sessionId: session.id,
        currentLine: session.debugger.currentLine,
        callstackDepth: callstack.length,
        callstack
      };
    },

    shader_debug_get_variables: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      return {
        sessionId: session.id,
        ...session.getVariables({
          scope: args.scope || "all",
          filter: args.filter || "",
          maxDepth: args.maxDepth !== undefined ? Number(args.maxDepth) : 2
        })
      };
    },

    shader_debug_eval: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const expr = args.expression || args.path;
      if (!expr || typeof expr !== "string") {
        throw new Error("expression or path is required.");
      }
      const session = sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(
          `No active shader debug session "${args.sessionId}". Use shader_debug_start first.`
        );
      }
      return {
        sessionId: session.id,
        ...session.evaluate(expr)
      };
    },

    shader_debug_stop: async (args) => {
      if (!args || !args.sessionId) {
        throw new Error("sessionId is required.");
      }
      const disposed = sessionManager.deleteSession(args.sessionId);
      return {
        sessionId: args.sessionId,
        disposed
      };
    }
  };

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = request.params.arguments || {};
    const handler = handlers[name];
    if (!handler) {
      return {
        content: [{ type: "text", text: `Unknown tool: ${name}` }],
        isError: true
      };
    }
    try {
      const result = await handler(args);
      // A handler may return raw MCP content items (e.g. an image screenshot)
      // that must not be JSON-stringified. Pass those through verbatim.
      if (result && Array.isArray(result.__mcpContent)) {
        return { content: result.__mcpContent };
      }
      const clamped = typeof result === "string" ? result : clampResult(result);
      const text =
        typeof clamped === "string" ? clamped : JSON.stringify(clamped, null, 2);
      return { content: [{ type: "text", text }] };
    } catch (e) {
      return {
        content: [
          {
            type: "text",
            text: `Error: ${e && e.message ? e.message : String(e)}`
          }
        ],
        isError: true
      };
    }
  });

  return server;
}
