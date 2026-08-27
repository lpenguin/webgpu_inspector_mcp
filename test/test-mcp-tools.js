#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import {
  createTestClient,
  createSampleCapture
} from "./mcp-harness.js";

const tests = [];

function test(name, fn) {
  tests.push({ name, fn });
}

/**
 * Creates a synthetic compute capture with a helper function, uniform, and storage buffer.
 */
function createComputeCapture() {
  const code = `
    struct Params {
      multiplier: f32,
      baseOffset: vec2f,
      mode: u32,
    };

    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage, read_write> outBuffer: array<vec4f>;

    fn compute_helper(v: f32, mult: f32) -> f32 {
      let offset = 5.0;
      let res = (v + offset) * mult;
      return res;
    }

    @compute @workgroup_size(1)
    fn comp_main(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      let rawX = f32(idx) + params.baseOffset.x;
      let scaledX = compute_helper(rawX, params.multiplier);
      let scaledY = (f32(idx) + params.baseOffset.y) * params.multiplier;
      outBuffer[idx] = vec4f(scaledX, scaledY, f32(params.mode), 1.0);
    }
  `;

  const uniformBuf = new ArrayBuffer(32);
  const f32U = new Float32Array(uniformBuf);
  const u32U = new Uint32Array(uniformBuf);
  f32U[0] = 2.0; // multiplier
  f32U[2] = 10.0; f32U[3] = 20.0; // baseOffset
  u32U[4] = 42; // mode

  const storageBuf = new ArrayBuffer(64);

  return {
    metadata: {
      frame: 1,
      commands: [
        { method: "createShaderModule", object: "GPUDevice#0", args: [{ label: "compute_shader" }], result: { __id: 1 } },
        { method: "createComputePipeline", object: "GPUDevice#0", args: [{ label: "compute_pipeline" }], result: { __id: 2 } },
        { method: "beginComputePass", object: "GPUCommandEncoder#1", args: [{ label: "compute_pass" }], result: "GPUComputePassEncoder#1" },
        { method: "setPipeline", object: "GPUComputePassEncoder#1", args: [{ __id: 2 }] },
        { method: "setBindGroup", object: "GPUComputePassEncoder#1", args: [0, { __id: 5 }] },
        { method: "dispatchWorkgroups", object: "GPUComputePassEncoder#1", args: [4, 1, 1] },
        { method: "end", object: "GPUComputePassEncoder#1", args: [] }
      ],
      objects: {
        "1": {
          id: 1,
          type: "ShaderModule",
          label: "compute_shader",
          descriptor: { code }
        },
        "2": {
          id: 2,
          type: "ComputePipeline",
          label: "compute_pipeline",
          descriptor: {
            compute: { module: { __id: 1 }, entryPoint: "comp_main" }
          }
        },
        "3": {
          id: 3,
          type: "Buffer",
          label: "params_uniform",
          size: 32,
          initialData: uniformBuf
        },
        "4": {
          id: 4,
          type: "Buffer",
          label: "output_storage",
          size: 64,
          initialData: storageBuf
        },
        "5": {
          id: 5,
          type: "BindGroup",
          label: "compute_bind_group",
          descriptor: {
            entries: [
              { binding: 0, resource: { buffer: { __id: 3 } } },
              { binding: 1, resource: { buffer: { __id: 4 } } }
            ]
          }
        }
      },
      validationErrors: []
    },
    payloads: {}
  };
}

// ---------------------------------------------------------------------------
// 1. Tool registration check
// ---------------------------------------------------------------------------
test("MCP server exposes all 8 shader debug tools in listTools", async () => {
  const harness = await createTestClient();
  try {
    const tools = await harness.listTools();
    const toolNames = new Set(tools.map((t) => t.name));

    const expectedShaderTools = [
      "shader_debug_start",
      "shader_debug_step",
      "shader_debug_continue",
      "shader_debug_set_breakpoints",
      "shader_debug_get_stack",
      "shader_debug_get_variables",
      "shader_debug_eval",
      "shader_debug_stop"
    ];

    for (const tool of expectedShaderTools) {
      assert(toolNames.has(tool), `Expected shader debug tool "${tool}" to be present`);
    }
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 2. shader_debug_start
// ---------------------------------------------------------------------------
test("shader_debug_start initializes compute debugging session and returns state snapshot", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      stage: "compute",
      invocation: { threadId: [1, 0, 0] }
    });

    assert(startRes.sessionId, "Must return sessionId");
    assert.equal(startRes.stage, "compute");
    assert.equal(startRes.entryPoint, "comp_main");
    assert.equal(startRes.status, "paused");
    assert(startRes.currentLine > 0, "Must have valid currentLine");
    assert(startRes.sourceSnippet.includes("->"), "Source snippet must contain active line indicator");
    assert.equal(startRes.callstackDepth, 1);
    assert.equal(startRes.isAtEnd, false);
    assert.deepEqual(startRes.invocation.threadId, [1, 0, 0]);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 3. shader_debug_step (step_next, step_into, step_over, step_out, count)
// ---------------------------------------------------------------------------
test("shader_debug_step advances execution across lines and handles helper stepping", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      invocation: { threadId: [0, 0, 0] }
    });
    const sessionId = startRes.sessionId;

    // 1. Single step_next (line 19 -> 20)
    const step1 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next"
    });
    assert.equal(step1.sessionId, sessionId);
    assert.equal(step1.status, "paused");
    assert.equal(step1.currentLine, 20);

    // 2. step_over (line 20 -> 21)
    const step2 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_over"
    });
    assert.equal(step2.currentLine, 21);

    // 3. Multi-step with count: 2 (line 21 -> 22 -> 23) using step_over
    const step3 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_over",
      count: 2
    });
    assert.equal(step3.currentLine, 23);

    // 4. Clean up
    await harness.callToolJson("shader_debug_stop", { sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 4. shader_debug_get_variables (all, locals, globals, filter, maxDepth)
// ---------------------------------------------------------------------------
test("shader_debug_get_variables inspects locals, globals, constants and supports filtering", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      invocation: { threadId: [2, 0, 0] }
    });
    const sessionId = startRes.sessionId;

    // Check globals (uniform params)
    const globalsRes = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      scope: "globals"
    });
    assert(globalsRes.globals, "globals object must be returned");
    assert.equal(globalsRes.globals.params.multiplier, 2.0);
    assert.deepEqual(globalsRes.globals.params.baseOffset, [10.0, 20.0]);
    assert.equal(globalsRes.globals.params.mode, 42);

    // Step through function to populate locals
    await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_over",
      count: 6
    });

    // Check locals
    const localsRes = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      scope: "locals"
    });
    assert(localsRes.locals, "locals object must be returned");
    assert.equal(localsRes.locals.idx, 2);
    assert.equal(localsRes.locals.rawX, 12); // 2 + 10

    // Check filter option
    const filteredRes = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      filter: "scaled"
    });
    assert(filteredRes.locals, "Filtered locals must be returned");
    assert("scaledX" in filteredRes.locals || "scaledY" in filteredRes.locals);
    assert(!("rawX" in filteredRes.locals));

    await harness.callToolJson("shader_debug_stop", { sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 5. shader_debug_eval (paths, struct access, array indexing, swizzles, errors)
// ---------------------------------------------------------------------------
test("shader_debug_eval evaluates path expressions, struct fields, indexing, and swizzling", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      invocation: { threadId: [3, 0, 0] }
    });
    const sessionId = startRes.sessionId;

    // Step to let locals define
    await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_over",
      count: 6
    });

    // 1. Scalar local
    const evalIdx = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "idx"
    });
    assert.equal(evalIdx.success, true);
    assert.equal(evalIdx.value, 3);

    // 2. Struct field
    const evalMult = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "params.multiplier"
    });
    assert.equal(evalMult.success, true);
    assert.equal(evalMult.value, 2.0);

    // 3. Struct vector field indexing
    const evalVecElem = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "params.baseOffset[1]"
    });
    assert.equal(evalVecElem.success, true);
    assert.equal(evalVecElem.value, 20.0);

    // 4. Vector swizzle
    const evalSwizzle = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "params.baseOffset.yx"
    });
    assert.equal(evalSwizzle.success, true);
    assert.deepEqual(evalSwizzle.value, [20.0, 10.0]);

    // 5. Non-existent variable error handling
    const evalMissing = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "nonExistentVar123"
    });
    assert.equal(evalMissing.success, false);
    assert(evalMissing.error.includes("not found"));

    await harness.callToolJson("shader_debug_stop", { sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 6. shader_debug_set_breakpoints and shader_debug_continue
// ---------------------------------------------------------------------------
test("shader_debug_set_breakpoints sets and clears breakpoints, shader_debug_continue stops at breakpoint", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      invocation: { threadId: [0, 0, 0] }
    });
    const sessionId = startRes.sessionId;

    // Set breakpoint on line 23: outBuffer[idx] = ...
    const bpRes = await harness.callToolJson("shader_debug_set_breakpoints", {
      sessionId,
      add: [23, 24]
    });
    assert.deepEqual(bpRes.activeBreakpoints, [23, 24]);

    // Remove line 24
    const bpRemoveRes = await harness.callToolJson("shader_debug_set_breakpoints", {
      sessionId,
      remove: [24]
    });
    assert.deepEqual(bpRemoveRes.activeBreakpoints, [23]);

    // Continue execution to hit line 23
    const contRes = await harness.callToolJson("shader_debug_continue", {
      sessionId
    });
    assert.equal(contRes.status, "paused");
    assert.equal(contRes.currentLine, 23);
    assert.equal(contRes.hitBreakpoint, 23);

    // Continue to finish
    const contFinish = await harness.callToolJson("shader_debug_continue", {
      sessionId
    });
    assert.equal(contFinish.status, "completed");
    assert.equal(contFinish.isAtEnd, true);

    await harness.callToolJson("shader_debug_stop", { sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 7. shader_debug_get_stack inside helper function
// ---------------------------------------------------------------------------
test("shader_debug_get_stack inspects callstack frames across helper function calls", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5,
      invocation: { threadId: [0, 0, 0] }
    });
    const sessionId = startRes.sessionId;

    // Step into compute_helper
    // Line 20: let idx = gid.x;
    // Line 21: let rawX = ...;
    // Line 22: let scaledX = compute_helper(rawX, params.multiplier);
    await harness.callToolJson("shader_debug_step", { sessionId, action: "step_next" }); // to line 21
    await harness.callToolJson("shader_debug_step", { sessionId, action: "step_next" }); // to line 22
    await harness.callToolJson("shader_debug_step", { sessionId, action: "step_into" }); // into helper

    const stackRes = await harness.callToolJson("shader_debug_get_stack", { sessionId });
    assert.equal(stackRes.sessionId, sessionId);
    assert(stackRes.callstackDepth >= 1, "Callstack depth should be at least 1");
    assert(Array.isArray(stackRes.callstack), "Callstack must be an array");

    const topFrame = stackRes.callstack[0];
    assert(topFrame.functionName === "compute_helper" || topFrame.functionName === "comp_main");

    await harness.callToolJson("shader_debug_stop", { sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 8. Vertex & Fragment debugging sessions over MCP
// ---------------------------------------------------------------------------
test("shader_debug_start on vertex and fragment shaders with vertex inputs and texture sampling", async () => {
  const harness = await createTestClient();
  try {
    const sampleCap = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sampleCap);

    // 1. Vertex Shader Session
    const vSession = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 6,
      stage: "vertex",
      invocation: { vertexIndex: 1 }
    });
    assert.equal(vSession.stage, "vertex");
    assert.equal(vSession.entryPoint, "vs_main");

    const vInputs = await harness.callToolJson("shader_debug_get_variables", {
      sessionId: vSession.sessionId,
      scope: "inputs"
    });
    assert(vInputs.inputs);
    assert.deepEqual(vInputs.inputs.pos || vInputs.inputs["0"], [-0.5, -0.5]);

    await harness.callToolJson("shader_debug_continue", { sessionId: vSession.sessionId });
    await harness.callToolJson("shader_debug_stop", { sessionId: vSession.sessionId });

    // 2. Fragment Shader Session
    const fSession = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 6,
      stage: "fragment",
      invocation: { pixelX: 100, pixelY: 100 }
    });
    assert.equal(fSession.stage, "fragment");
    assert.equal(fSession.entryPoint, "fs_main");

    const fStep = await harness.callToolJson("shader_debug_continue", { sessionId: fSession.sessionId });
    assert.equal(fStep.status, "completed");

    await harness.callToolJson("shader_debug_stop", { sessionId: fSession.sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 9. shader_debug_stop and invalid session error handling
// ---------------------------------------------------------------------------
test("shader_debug_stop disposes session and rejects subsequent operations on invalid sessionId", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 5
    });
    const sessionId = startRes.sessionId;

    const stopRes = await harness.callToolJson("shader_debug_stop", { sessionId });
    assert.equal(stopRes.sessionId, sessionId);
    assert.equal(stopRes.disposed, true);

    // Subsequent step call should fail
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_step", { sessionId });
      },
      (err) => err.message.includes(`No active shader debug session "${sessionId}"`)
    );
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 10. Validation against WoodWorks real capture if present
// ---------------------------------------------------------------------------
test("Validation against real WoodWorks capture file (if present on filesystem)", async () => {
  const possiblePaths = [
    path.resolve(process.cwd(), "../WoodWorks/webgpu_capture_frame_3935.wgpuc"),
    path.resolve("/home/nikita/WoodWorks/webgpu_capture_frame_3935.wgpuc"),
    path.resolve(process.cwd(), "captures/webgpu_capture_frame_3935.wgpuc")
  ];

  let capturePath = null;
  for (const p of possiblePaths) {
    try {
      await fs.access(p);
      capturePath = p;
      break;
    } catch {}
  }

  if (!capturePath) {
    console.log("    (Skipping WoodWorks real capture test: file not found)");
    return;
  }

  const harness = await createTestClient();
  try {
    // 1. Load capture via load_capture_file tool
    const loadRes = await harness.callToolJson("load_capture_file", { path: capturePath });
    assert(loadRes.captureId, "Must return loaded captureId");
    assert.equal(loadRes.summary.totalObjects, 100);

    const captureId = loadRes.captureId;

    // 2. Fetch ShaderModule 93 (Shaving evaluate TypeGPU)
    const shaderModule = await harness.callToolJson("get_shader", {
      captureId,
      objectId: 93
    });
    assert(shaderModule.code.includes("@compute"), "ShaderModule 93 must contain compute kernel");
    assert(shaderModule.code.includes("evaluateShaving"), "Must contain evaluateShaving entry point");

    // 3. Start debugging session on ShaderModule 93 compute shader
    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId,
      code: shaderModule.code,
      stage: "compute",
      entryPoint: "evaluateShaving",
      invocation: { threadId: [0, 0, 0] }
    });

    assert(startRes.sessionId, "Must return sessionId for WoodWorks compute shader");
    assert.equal(startRes.stage, "compute");
    assert.equal(startRes.entryPoint, "evaluateShaving");
    assert.equal(startRes.status, "paused");

    const sessionId = startRes.sessionId;

    // 4. Step through first few lines
    const step1 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next",
      count: 3
    });
    assert.equal(step1.status, "paused");
    assert(step1.currentLine > 0);

    // 5. Inspect variables
    const vars = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      scope: "locals"
    });
    assert(vars.locals, "Locals should be returned");

    // 6. Evaluate local variable
    const evalRes = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "shaving"
    });
    assert.equal(evalRes.success, true);

    // 7. Get callstack
    const stack = await harness.callToolJson("shader_debug_get_stack", { sessionId });
    assert.equal(stack.callstackDepth, 1);
    assert.equal(stack.callstack[0].functionName, "evaluateShaving");

    // 8. Stop session
    const stopRes = await harness.callToolJson("shader_debug_stop", { sessionId });
    assert.equal(stopRes.disposed, true);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 11. Referencing shaders across captures and by object ID
// ---------------------------------------------------------------------------
test("shader_debug_start references shader code from another capture via shaderCaptureId and shaderObjectId", async () => {
  const harness = await createTestClient();
  try {
    // Capture A: contains ShaderModule #10 with WGSL compute code
    const shaderCode = `
      struct Params {
        multiplier: f32,
      };
      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read_write> outBuffer: array<f32>;

      @compute @workgroup_size(1)
      fn cross_compute(@builtin(global_invocation_id) gid: vec3u) {
        let idx = gid.x;
        outBuffer[idx] = f32(idx) * params.multiplier + 10.0;
      }
    `;

    const captureA = {
      metadata: {
        frame: 1,
        commands: [],
        objects: {
          "10": {
            id: 10,
            type: "ShaderModule",
            label: "external_compute_module",
            descriptor: { code: shaderCode }
          }
        }
      }
    };

    // Capture B: contains pipeline without shader source in its objects, but has uniform & storage buffers
    const uniformBuf = new ArrayBuffer(16);
    new Float32Array(uniformBuf)[0] = 3.5;
    const storageBuf = new ArrayBuffer(64);

    const captureB = {
      metadata: {
        frame: 2,
        commands: [
          { method: "beginComputePass", object: "GPUCommandEncoder#1", args: [{ label: "compute_pass" }], result: "GPUComputePassEncoder#1" },
          { method: "setPipeline", object: "GPUComputePassEncoder#1", args: [{ __id: 20 }] },
          { method: "setBindGroup", object: "GPUComputePassEncoder#1", args: [0, { __id: 25 }] },
          { method: "dispatchWorkgroups", object: "GPUComputePassEncoder#1", args: [4, 1, 1] },
          { method: "end", object: "GPUComputePassEncoder#1", args: [] }
        ],
        objects: {
          "20": {
            id: 20,
            type: "ComputePipeline",
            label: "target_pipeline",
            descriptor: {
              compute: { module: { __id: 10 }, entryPoint: "cross_compute" }
            }
          },
          "23": {
            id: 23,
            type: "Buffer",
            label: "params_uniform",
            size: 16,
            initialData: uniformBuf
          },
          "24": {
            id: 24,
            type: "Buffer",
            label: "output_storage",
            size: 64,
            initialData: storageBuf
          },
          "25": {
            id: 25,
            type: "BindGroup",
            label: "target_bind_group",
            descriptor: {
              entries: [
                { binding: 0, resource: { buffer: { __id: 23 }, offset: 0, size: 16 } },
                { binding: 1, resource: { buffer: { __id: 24 }, offset: 0, size: 64 } }
              ]
            }
          }
        }
      }
    };

    const metaA = await harness.loadSyntheticCapture(captureA, { label: "capture-a" });
    const metaB = await harness.loadSyntheticCapture(captureB, { label: "capture-b" });

    // Start debug session on capture B referencing shader module #10 from capture A
    const session = await harness.callToolJson("shader_debug_start", {
      captureId: metaB.id,
      shaderCaptureId: metaA.id,
      shaderObjectId: 10,
      commandIndex: 3,
      invocation: { threadId: [1, 0, 0] }
    });

    assert.equal(session.captureId, metaB.id);
    assert.equal(session.stage, "compute");
    assert.equal(session.entryPoint, "cross_compute");
    assert.equal(session.status, "paused");

    // Step through execution
    const stepRes = await harness.callToolJson("shader_debug_step", {
      sessionId: session.sessionId,
      action: "step_next",
      count: 1
    });
    assert.equal(stepRes.status, "paused");

    // Inspect variables
    const vars = await harness.callToolJson("shader_debug_get_variables", {
      sessionId: session.sessionId,
      scope: "globals"
    });
    assert(vars.globals.params, "params uniform buffer must be resolved in globals");

    // Evaluate expression
    const evalRes = await harness.callToolJson("shader_debug_eval", {
      sessionId: session.sessionId,
      expression: "params.multiplier"
    });
    assert.equal(evalRes.success, true);
    assert.equal(evalRes.value, 3.5);

    await harness.callToolJson("shader_debug_stop", { sessionId: session.sessionId });
  } finally {
    await harness.close();
  }
});

test("shader_debug_start references shaderObjectId within same capture", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    // Call shader_debug_start referencing shaderObjectId: 1 without passing code or shaderCaptureId
    const session = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      shaderObjectId: 1,
      commandIndex: 5,
      invocation: { threadId: [0, 0, 0] }
    });

    assert.equal(session.captureId, meta.id);
    assert.equal(session.stage, "compute");
    assert.equal(session.entryPoint, "comp_main");
    assert.equal(session.status, "paused");

    const stepRes = await harness.callToolJson("shader_debug_step", {
      sessionId: session.sessionId,
      action: "step_next",
      count: 2
    });
    assert.equal(stepRes.status, "paused");

    await harness.callToolJson("shader_debug_stop", { sessionId: session.sessionId });
  } finally {
    await harness.close();
  }
});

test("shader_debug_start error handling for shaderCaptureId and shaderObjectId", async () => {
  const harness = await createTestClient();
  try {
    const computeCap = createComputeCapture();
    const meta = await harness.loadSyntheticCapture(computeCap);

    // 1. Non-existent shaderCaptureId
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_start", {
          captureId: meta.id,
          shaderCaptureId: "non-existent-cap-999",
          shaderObjectId: 1
        });
      },
      (err) => err.message.includes('No capture "non-existent-cap-999"')
    );

    // 2. Non-existent shaderObjectId
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_start", {
          captureId: meta.id,
          shaderObjectId: 9999
        });
      },
      (err) => err.message.includes("No object #9999 in capture")
    );

    // 3. shaderCaptureId specified without shaderObjectId
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_start", {
          captureId: meta.id,
          shaderCaptureId: meta.id
        });
      },
      (err) => err.message.includes("shaderObjectId is required when shaderCaptureId is specified")
    );

    // 4. shaderObjectId pointing to a Buffer (not a ShaderModule)
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_start", {
          captureId: meta.id,
          shaderObjectId: 3
        });
      },
      (err) => err.message.includes("Object #3 is a Buffer, not a ShaderModule")
    );

    // 5. shaderObjectId pointing to ShaderModule with empty code
    const emptyShaderCap = {
      metadata: {
        frame: 1,
        commands: [],
        objects: {
          "1": {
            id: 1,
            type: "ShaderModule",
            label: "empty_shader",
            descriptor: { code: "" }
          }
        }
      }
    };
    const metaEmpty = await harness.loadSyntheticCapture(emptyShaderCap);
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_start", {
          captureId: metaEmpty.id,
          shaderObjectId: 1
        });
      },
      (err) => err.message.includes("Shader module #1 contains no WGSL code")
    );

    // 6. Explicit 'code' override takes precedence over shaderObjectId
    const overrideCode = `
      @compute @workgroup_size(1)
      fn override_main(@builtin(global_invocation_id) gid: vec3u) {}
    `;
    const sessionOverride = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      code: overrideCode,
      shaderObjectId: 9999, // Should be ignored because 'code' is provided
      commandIndex: 5,
      invocation: { threadId: [0, 0, 0] }
    });
    assert.equal(sessionOverride.entryPoint, "override_main");
    await harness.callToolJson("shader_debug_stop", { sessionId: sessionOverride.sessionId });
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------
async function runAll() {
  let passed = 0;
  let failed = 0;
  const startTime = Date.now();

  console.log(`\nRunning ${tests.length} Shader Debug MCP Tools Tests...\n`);

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
