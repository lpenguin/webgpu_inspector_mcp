#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  createTestServer,
  createTestClient,
  createSampleCapture
} from "./mcp-harness.js";

const tests = [];

function test(name, fn) {
  tests.push({ name, fn });
}

test("MCP Client connects and lists all required WebGPU Inspector tools", async () => {
  const harness = await createTestClient();
  try {
    const tools = await harness.listTools();
    assert(Array.isArray(tools), "listTools should return an array of tools");
    assert(tools.length >= 16, `Expected at least 16 tools, got ${tools.length}`);

    const toolNames = new Set(tools.map((t) => t.name));
    const expectedTools = [
      "launch_browser",
      "attach_browser",
      "open_page",
      "browser_status",
      "list_pages",
      "screenshot_page",
      "capture_frames",
      "list_captures",
      "load_capture_file",
      "get_capture_summary",
      "analyze_performance",
      "get_commands",
      "get_object",
      "get_shader",
      "get_validation_errors",
      "get_draw_state",
      "decode_vertex_buffer",
      "diff_draws",
      "read_buffer",
      "read_texture",
      "get_frame_stats",
      "shader_debug_start",
      "shader_debug_step",
      "shader_debug_continue",
      "shader_debug_set_breakpoints",
      "shader_debug_get_stack",
      "shader_debug_get_variables",
      "shader_debug_eval",
      "shader_debug_stop"
    ];

    for (const tool of expectedTools) {
      assert(toolNames.has(tool), `Expected tool "${tool}" to be registered on server`);
    }
  } finally {
    await harness.close();
  }
});

test("list_captures returns empty list initially", async () => {
  const harness = await createTestClient();
  try {
    const res = await harness.callToolJson("list_captures");
    assert(res && Array.isArray(res.captures), "Should return captures array");
    assert.equal(res.captures.length, 0, "Initial captures should be empty");
  } finally {
    await harness.close();
  }
});

test("loadSyntheticCapture seeds CaptureStore and list_captures reflects it", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample, { label: "test-triangle-capture" });

    assert.equal(typeof meta.id, "string", "Should return capture ID");
    assert.equal(meta.label, "test-triangle-capture");
    assert.equal(meta.totalCommands, 9);
    assert.equal(meta.totalObjects, 6);

    const listRes = await harness.callToolJson("list_captures");
    assert.equal(listRes.captures.length, 1);
    assert.equal(listRes.captures[0].id, meta.id);
  } finally {
    await harness.close();
  }
});

test("get_capture_summary returns pass breakdown and object statistics", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("get_capture_summary", { captureId: meta.id });
    assert.equal(res.captureId, meta.id);
    assert(res.summary, "Summary object must be present");
    assert.equal(res.summary.totalCommands, 9);
    assert.equal(res.summary.totalObjects, 6);
    assert.equal(res.summary.objectCounts.ShaderModule, 1);
    assert.equal(res.summary.objectCounts.RenderPipeline, 1);

    assert(Array.isArray(res.summary.passes), "Passes should be an array");
    assert.equal(res.summary.passes.length, 1);
    assert.equal(res.summary.passes[0].label, "main_render_pass");
    assert.equal(res.summary.passes[0].draws, 2);
    assert.equal(res.summary.passes[0].durationMs, 0.85);
  } finally {
    await harness.close();
  }
});

test("get_commands supports pagination and method filtering", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    // All commands
    const allCmds = await harness.callToolJson("get_commands", { captureId: meta.id });
    assert.equal(allCmds.total, 9);
    assert.equal(allCmds.commands.length, 9);

    // Filter by draw
    const drawCmds = await harness.callToolJson("get_commands", {
      captureId: meta.id,
      method: "draw"
    });
    assert.equal(drawCmds.total, 2);
    assert.equal(drawCmds.commands.length, 2);
    assert.equal(drawCmds.commands[0].method, "draw");
    assert.equal(drawCmds.commands[1].method, "draw");

    // Pagination
    const page = await harness.callToolJson("get_commands", {
      captureId: meta.id,
      offset: 2,
      limit: 3
    });
    assert.equal(page.offset, 2);
    assert.equal(page.limit, 3);
    assert.equal(page.commands.length, 3);
    assert.equal(page.commands[0].index, 2);
  } finally {
    await harness.close();
  }
});

test("get_object retrieves stripped GPU object records", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const shaderObj = await harness.callToolJson("get_object", {
      captureId: meta.id,
      objectId: 1
    });
    assert.equal(shaderObj.object.id, 1);
    assert.equal(shaderObj.object.type, "ShaderModule");
    assert.equal(shaderObj.object.label, "triangle_shader");

    const pipelineObj = await harness.callToolJson("get_object", {
      captureId: meta.id,
      objectId: 2
    });
    assert.equal(pipelineObj.object.id, 2);
    assert.equal(pipelineObj.object.type, "RenderPipeline");
    assert.equal(pipelineObj.object.label, "triangle_pipeline");
  } finally {
    await harness.close();
  }
});

test("get_shader retrieves WGSL shader source code", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("get_shader", {
      captureId: meta.id,
      objectId: 1
    });
    assert.equal(res.id, 1);
    assert.equal(res.label, "triangle_shader");
    assert.equal(res.hasVertexEntries, true);
    assert.equal(res.hasFragmentEntries, true);
    assert(typeof res.code === "string", "Code should be string");
    assert(res.code.includes("@vertex"), "WGSL code should contain vertex entry");
    assert(res.code.includes("@fragment"), "WGSL code should contain fragment entry");
  } finally {
    await harness.close();
  }
});

test("get_draw_state resolves pipeline, vertex buffers, and bind groups for draw call", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("get_draw_state", {
      captureId: meta.id,
      commandIndex: 6
    });

    assert.equal(res.captureId, meta.id);
    const ds = res.drawState;
    assert.equal(ds.commandIndex, 6);
    assert.equal(ds.method, "draw");
    assert.deepEqual(ds.drawArgs, [3, 1, 0, 0]);

    // Pipeline resolved
    assert(ds.pipeline, "Pipeline should be resolved");
    assert.equal(ds.pipeline.id, 2);
    assert.equal(ds.pipeline.label, "triangle_pipeline");

    // Vertex buffers resolved
    assert(Array.isArray(ds.vertexBuffers), "vertexBuffers should be array");
    assert.equal(ds.vertexBuffers.length, 1);
    assert.equal(ds.vertexBuffers[0].slot, 0);
    assert.equal(ds.vertexBuffers[0].bufferId, 3);
    assert.equal(ds.vertexBuffers[0].bufferDataCommandIndex, 5);
    assert(ds.vertexBuffers[0].layout, "Vertex buffer layout should be resolved");
    assert.equal(ds.vertexBuffers[0].layout.arrayStride, 8);

    // Bind groups resolved
    assert(Array.isArray(ds.bindGroups), "bindGroups should be array");
    assert.equal(ds.bindGroups.length, 1);
    assert.equal(ds.bindGroups[0].group, 0);
    assert.equal(ds.bindGroups[0].bindGroupId, 6);
  } finally {
    await harness.close();
  }
});

test("decode_vertex_buffer decodes vertex buffer payload into typed attribute values", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("decode_vertex_buffer", {
      captureId: meta.id,
      commandIndex: 5,
      firstN: 3
    });

    assert.equal(res.captureId, meta.id);
    assert.equal(res.slot, 0);
    assert.equal(res.arrayStride, 8);
    assert.equal(res.vertexCount, 3);
    assert.equal(res.vertices.length, 3);

    // Vertex 0: (0.0, 0.5)
    assert.equal(res.vertices[0].vertex, 0);
    assert.deepEqual(res.vertices[0].attributes["0"].value, [0, 0.5]);

    // Vertex 1: (-0.5, -0.5)
    assert.equal(res.vertices[1].vertex, 1);
    assert.deepEqual(res.vertices[1].attributes["0"].value, [-0.5, -0.5]);

    // Vertex 2: (0.5, -0.5)
    assert.equal(res.vertices[2].vertex, 2);
    assert.deepEqual(res.vertices[2].attributes["0"].value, [0.5, -0.5]);
  } finally {
    await harness.close();
  }
});

test("diff_draws compares two draw calls and detects identical state", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("diff_draws", {
      captureId: meta.id,
      cmdA: 6,
      cmdB: 7
    });

    assert.equal(res.captureId, meta.id);
    assert.equal(res.identical, true);
    assert.deepEqual(res.differences, []);
  } finally {
    await harness.close();
  }
});

test("get_validation_errors returns empty list for valid capture", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("get_validation_errors", {
      captureId: meta.id
    });
    assert.equal(res.captureId, meta.id);
    assert(Array.isArray(res.validationErrors));
    assert.equal(res.validationErrors.length, 0);
  } finally {
    await harness.close();
  }
});

test("analyze_performance evaluates render pass targets and durations", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    const res = await harness.callToolJson("analyze_performance", {
      captureId: meta.id
    });

    assert.equal(res.captureId, meta.id);
    assert(res.performance, "performance property must be present");
    assert(Array.isArray(res.performance.passes));
    assert.equal(res.performance.passes.length, 1);
    assert.equal(res.performance.passes[0].label, "main_render_pass");
    assert(res.performance.passes[0].target, "target info should be present");
    assert.equal(res.performance.passes[0].target.width, 800);
    assert.equal(res.performance.passes[0].target.height, 600);
  } finally {
    await harness.close();
  }
});

test("load_capture_file loads a capture file from disk", async () => {
  const tmpDir = path.join(os.tmpdir(), `webgpu-file-test-${Date.now()}`);
  await fs.mkdir(tmpDir, { recursive: true });
  const filePath = path.join(tmpDir, "test-capture.json");

  const sample = createSampleCapture();
  await fs.writeFile(filePath, JSON.stringify(sample.metadata, null, 2), "utf8");

  const harness = await createTestClient();
  try {
    const res = await harness.callToolJson("load_capture_file", { path: filePath });
    assert(res.captureId, "Should return captureId");
    assert(res.summary, "Should return summary");
    assert.equal(res.summary.totalCommands, 9);

    const listRes = await harness.callToolJson("list_captures");
    assert.equal(listRes.captures.length, 1);
    assert.equal(listRes.captures[0].id, res.captureId);
  } finally {
    await harness.close();
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
});

test("callToolJson returns raw tool error structure or throws on failed call", async () => {
  const harness = await createTestClient();
  try {
    const sample = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sample);

    // Request non-existent object
    await assert.rejects(
      async () => {
        await harness.callToolJson("get_object", {
          captureId: meta.id,
          objectId: 9999
        });
      },
      (err) => {
        return err.message.includes("No object #9999");
      }
    );
  } finally {
    await harness.close();
  }
});

// Runner
async function runAll() {
  let passed = 0;
  let failed = 0;
  const startTime = Date.now();

  console.log(`\nRunning ${tests.length} WebGPU Inspector MCP tests...\n`);

  for (const { name, fn } of tests) {
    try {
      await fn();
      console.log(`  ✓ ${name}`);
      passed++;
    } catch (err) {
      console.error(`  ✗ ${name}`);
      console.error(`    ${err.stack || err.message}\n`);
      failed++;
    }
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`\n--------------------------------------------------`);
  console.log(`Results: ${passed} passed, ${failed} failed (${duration}s)`);

  if (failed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

runAll().catch((err) => {
  console.error("Fatal test runner error:", err);
  process.exit(1);
});
