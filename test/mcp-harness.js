import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMcpServer } from "../src/mcp-server.js";
import { CaptureStore } from "webgpu_inspector/claude-plugin/server/capture-store.js";

/**
 * Creates an in-process MCP Server configured for testing WebGPU Inspector tools.
 *
 * @param {Object} [options]
 * @param {string} [options.capturesDir] - Directory for storing captures
 * @param {CaptureStore} [options.store] - Optional custom CaptureStore
 * @param {Object} [options.bridge] - Optional custom bridge
 * @param {Object} [options.browser] - Optional custom browser controller
 * @param {string} [options.version] - Server version string
 * @param {boolean} [options.cleanDir] - Clean capturesDir on dispose
 * @returns {Promise<{ server: any, store: CaptureStore, bridge: any, browser: any, capturesDir: string, close: Function, dispose: Function }>}
 */
export async function createTestServer(options = {}) {
  const isGeneratedDir = !options.capturesDir && !options.dir;
  const capturesDir = options.capturesDir || options.dir || path.join(
    os.tmpdir(),
    `webgpu-mcp-test-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
  const cleanDir = options.cleanDir ?? isGeneratedDir;

  let store = options.store;
  if (!store) {
    store = new CaptureStore({ dir: capturesDir });
    await store.init();
  }

  const defaultBridge = {
    isListening: () => false,
    listPages: () => [],
    waitForPage: async () => null,
    pageInstanceId: () => null,
    requestCapture: async () => ({ id: "cap-1" }),
    requestRead: async () => ({}),
    requestReadTexture: async () => ({}),
    requestFrameStats: async () => ({})
  };

  const defaultBrowser = {
    status: () => ({ isConnected: false, targets: [] }),
    dispose: async () => {},
    launch: async () => ({ opened: null }),
    attach: async () => ({ isConnected: false }),
    openPage: async () => ({ url: "", instanceId: "" }),
    screenshot: async () => ({ base64: "", mimeType: "image/png" })
  };

  const bridge = options.bridge || defaultBridge;
  const browser = options.browser || defaultBrowser;
  const sessionManager = options.sessionManager;
  const version = options.version || "0.1.0";

  const server = createMcpServer({
    store,
    bridge,
    browser,
    sessionManager,
    version
  });

  const dispose = async () => {
    try {
      if (server.close) {
        await server.close();
      }
    } catch {}
    if (cleanDir && capturesDir.startsWith(os.tmpdir())) {
      try {
        await fs.rm(capturesDir, { recursive: true, force: true });
      } catch {}
    }
  };

  return {
    server,
    store,
    bridge,
    browser,
    sessionManager: server.sessionManager,
    capturesDir,
    close: dispose,
    dispose
  };
}

/**
 * Creates an in-process MCP Client linked to a test server via InMemoryTransport.
 *
 * @param {Object|Server} [serverInstance] - Server instance or test server wrapper
 * @param {Object} [options] - Options if serverInstance is auto-created
 * @returns {Promise<{ client: Client, server: any, store: CaptureStore, listTools: Function, callTool: Function, callToolJson: Function, loadSyntheticCapture: Function, close: Function, dispose: Function }>}
 */
export async function createTestClient(serverInstance, options = {}) {
  let serverWrapper = null;
  let server = null;
  let store = null;

  if (!serverInstance || (!serverInstance.connect && !serverInstance.server)) {
    serverWrapper = await createTestServer(serverInstance || options);
    server = serverWrapper.server;
    store = serverWrapper.store;
  } else if (serverInstance.server) {
    serverWrapper = serverInstance;
    server = serverInstance.server;
    store = serverInstance.store;
  } else {
    server = serverInstance;
    store = options.store || null;
  }

  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  await server.connect(serverTransport);

  const client = new Client(
    { name: "webgpu-inspector-test-client", version: "1.0.0" },
    { capabilities: {} }
  );

  await client.connect(clientTransport);

  const harness = {
    client,
    server,
    store,
    sessionManager: server.sessionManager,
    clientTransport,
    serverTransport,

    async listTools() {
      const result = await client.listTools();
      return result.tools || result;
    },

    async callTool(name, args = {}) {
      return await client.callTool({
        name,
        arguments: args
      });
    },

    async callToolJson(name, args = {}) {
      const result = await client.callTool({
        name,
        arguments: args
      });

      if (result.isError) {
        const text = result.content?.[0]?.text || `Tool "${name}" returned an error`;
        const err = new Error(text);
        err.isError = true;
        err.raw = result;
        try {
          err.data = JSON.parse(text);
        } catch {}
        throw err;
      }

      const textItem = result.content?.find((c) => c.type === "text");
      if (!textItem || typeof textItem.text !== "string") {
        throw new Error(`Tool "${name}" returned no text content in MCP result.`);
      }

      return JSON.parse(textItem.text);
    },

    async loadSyntheticCapture(captureData, meta = {}) {
      if (!store) {
        throw new Error("No CaptureStore available to seed synthetic capture.");
      }
      const rawJson = captureData.metadata || captureData;
      const rawPayloads = captureData.payloads;
      const payloads = new Map();

      if (rawPayloads instanceof Map) {
        for (const [k, v] of rawPayloads.entries()) {
          payloads.set(k, v);
        }
      } else if (rawPayloads && typeof rawPayloads === "object") {
        for (const [k, v] of Object.entries(rawPayloads)) {
          const id = Number(k) || k;
          if (Buffer.isBuffer(v) || v instanceof Uint8Array) {
            payloads.set(id, { bytes: Buffer.from(v) });
          } else if (typeof v === "string") {
            payloads.set(id, { base64: v, bytes: Buffer.from(v, "base64") });
          } else {
            payloads.set(id, v);
          }
        }
      }

      return await store.addLive({ metadata: rawJson, payloads }, meta);
    },

    async close() {
      try {
        await client.close();
      } catch {}
      try {
        await clientTransport.close();
      } catch {}
      try {
        await serverTransport.close();
      } catch {}
      if (serverWrapper?.close) {
        try {
          await serverWrapper.close();
        } catch {}
      }
    },

    async dispose() {
      return await this.close();
    }
  };

  return harness;
}

/**
 * Creates a valid synthetic triangle rendering WebGPU capture for tests.
 *
 * @param {Object} [overrides]
 * @returns {{ metadata: Object, payloads: Object }}
 */
export function createSampleCapture(overrides = {}) {
  const vertexFloats = new Float32Array([
    0.0, 0.5,
    -0.5, -0.5,
    0.5, -0.5
  ]);
  const vertexBufferBytes = Buffer.from(vertexFloats.buffer);

  const capture = {
    metadata: {
      frame: 1,
      commands: [
        {
          method: "createShaderModule",
          object: "GPUDevice#0",
          args: [{ label: "triangle_shader" }],
          result: { __id: 1 }
        },
        {
          method: "createRenderPipeline",
          object: "GPUDevice#0",
          args: [{ label: "triangle_pipeline" }],
          result: { __id: 2 }
        },
        {
          method: "beginRenderPass",
          object: "GPUCommandEncoder#1",
          args: [
            {
              label: "main_render_pass",
              colorAttachments: [
                {
                  view: { __id: 5 },
                  loadOp: "clear",
                  storeOp: "store"
                }
              ]
            }
          ],
          result: "GPURenderPassEncoder#2",
          duration: 0.85
        },
        {
          method: "setPipeline",
          object: "GPURenderPassEncoder#2",
          args: [{ __id: 2 }]
        },
        {
          method: "setBindGroup",
          object: "GPURenderPassEncoder#2",
          args: [0, { __id: 6 }]
        },
        {
          method: "setVertexBuffer",
          object: "GPURenderPassEncoder#2",
          args: [0, { __id: 3 }, 0, 24],
          bufferData: [
            {
              entryIndex: 0,
              __payloadId: 101,
              __typedArray: "Float32Array",
              __length: 6,
              __byteLength: 24
            }
          ]
        },
        {
          method: "draw",
          object: "GPURenderPassEncoder#2",
          args: [3, 1, 0, 0]
        },
        {
          method: "draw",
          object: "GPURenderPassEncoder#2",
          args: [3, 1, 0, 0]
        },
        {
          method: "end",
          object: "GPURenderPassEncoder#2",
          args: []
        }
      ],
      objects: {
        "1": {
          id: 1,
          type: "ShaderModule",
          label: "triangle_shader",
          hasVertexEntries: true,
          hasFragmentEntries: true,
          hasComputeEntries: false,
          descriptor: {
            code: `@vertex\nfn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {\n  return vec4f(pos, 0.0, 1.0);\n}\n\n@fragment\nfn fs_main() -> @location(0) vec4f {\n  return vec4f(1.0, 0.0, 0.0, 1.0);\n}`
          }
        },
        "2": {
          id: 2,
          type: "RenderPipeline",
          label: "triangle_pipeline",
          descriptor: {
            vertex: {
              module: { __id: 1 },
              entryPoint: "vs_main",
              buffers: [
                {
                  arrayStride: 8,
                  attributes: [
                    { shaderLocation: 0, offset: 0, format: "float32x2" }
                  ]
                }
              ]
            },
            fragment: {
              module: { __id: 1 },
              entryPoint: "fs_main",
              targets: [{ format: "rgba8unorm" }]
            }
          }
        },
        "3": {
          id: 3,
          type: "Buffer",
          label: "vertex_buffer",
          size: 24,
          usage: 32
        },
        "4": {
          id: 4,
          type: "Texture",
          label: "color_target",
          width: 800,
          height: 600,
          format: "rgba8unorm",
          descriptor: {
            size: [800, 600, 1],
            format: "rgba8unorm",
            sampleCount: 1,
            usage: 16
          }
        },
        "5": {
          id: 5,
          type: "TextureView",
          label: "color_target_view",
          texture: { __id: 4 }
        },
        "6": {
          id: 6,
          type: "BindGroup",
          label: "main_bind_group",
          descriptor: {
            entries: []
          }
        }
      },
      validationErrors: []
    },
    payloads: {
      101: {
        __typedArray: "Float32Array",
        bytes: vertexBufferBytes,
        base64: vertexBufferBytes.toString("base64")
      }
    }
  };

  if (overrides.metadata) {
    Object.assign(capture.metadata, overrides.metadata);
  }
  if (overrides.payloads) {
    Object.assign(capture.payloads, overrides.payloads);
  }
  return capture;
}
