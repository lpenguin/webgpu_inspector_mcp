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

// ---------------------------------------------------------------------------
// Helper to locate real WoodWorks capture file
// ---------------------------------------------------------------------------
async function findWoodWorksCapturePath() {
  const possiblePaths = [
    path.resolve(process.cwd(), "../WoodWorks/webgpu_capture_frame_3935.wgpuc"),
    path.resolve("/home/nikita/WoodWorks/webgpu_capture_frame_3935.wgpuc"),
    path.resolve(process.cwd(), "captures/webgpu_capture_frame_3935.wgpuc")
  ];

  for (const p of possiblePaths) {
    try {
      await fs.access(p);
      return p;
    } catch {}
  }
  return null;
}

// ---------------------------------------------------------------------------
// 1. Concurrent Debug Sessions on the Same MCP Server
// ---------------------------------------------------------------------------
test("Concurrent debug sessions: vertex, fragment, and compute sessions running in parallel", async () => {
  const harness = await createTestClient();
  try {
    const sampleCap = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sampleCap);

    // Compute Shader Code
    const computeCode = `
      struct Data { val: f32, mult: f32 };
      @group(0) @binding(0) var<uniform> uData: Data;
      @compute @workgroup_size(1)
      fn comp_main(@builtin(global_invocation_id) gid: vec3u) {
        let x = f32(gid.x) + uData.val;
        let y = x * uData.mult;
      }
    `;

    // 1. Start Vertex Debug Session (Session A)
    const sessionA = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 6,
      stage: "vertex",
      invocation: { vertexIndex: 1 }
    });
    assert(sessionA.sessionId, "Session A must have sessionId");
    assert.equal(sessionA.stage, "vertex");

    // 2. Start Fragment Debug Session (Session B)
    const sessionB = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 6,
      stage: "fragment",
      invocation: { pixelX: 100, pixelY: 100 }
    });
    assert(sessionB.sessionId, "Session B must have sessionId");
    assert.equal(sessionB.stage, "fragment");
    assert.notEqual(sessionA.sessionId, sessionB.sessionId, "Session IDs must be distinct");

    // 3. Start Compute Debug Session (Session C)
    const sessionC = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      code: computeCode,
      stage: "compute",
      entryPoint: "comp_main",
      invocation: { threadId: [4, 0, 0] }
    });
    assert(sessionC.sessionId, "Session C must have sessionId");
    assert.equal(sessionC.stage, "compute");

    // Verify all 3 sessions are tracked simultaneously
    const listSessions = harness.sessionManager.listSessions();
    assert.equal(listSessions.length, 3);
    const activeIds = listSessions.map((s) => s.sessionId);
    assert(activeIds.includes(sessionA.sessionId));
    assert(activeIds.includes(sessionB.sessionId));
    assert(activeIds.includes(sessionC.sessionId));

    // 4. Interleaved operations on concurrent sessions
    // Step Session A
    const stepA = await harness.callToolJson("shader_debug_step", {
      sessionId: sessionA.sessionId,
      action: "step_next"
    });
    assert.equal(stepA.sessionId, sessionA.sessionId);

    // Set breakpoint and continue on Session C
    await harness.callToolJson("shader_debug_set_breakpoints", {
      sessionId: sessionC.sessionId,
      add: [7] // let y = x * uData.mult;
    });

    const contC = await harness.callToolJson("shader_debug_continue", {
      sessionId: sessionC.sessionId
    });
    assert.equal(contC.sessionId, sessionC.sessionId);
    assert.equal(contC.hitBreakpoint, 7);

    // Inspect Session A variables (should remain vertex inputs, unaffected by Session C)
    const varsA = await harness.callToolJson("shader_debug_get_variables", {
      sessionId: sessionA.sessionId,
      scope: "inputs"
    });
    assert.deepEqual(varsA.inputs.pos || varsA.inputs["0"], [-0.5, -0.5]);

    // Inspect Session C variables (should have compute gid and local x)
    const varsC = await harness.callToolJson("shader_debug_get_variables", {
      sessionId: sessionC.sessionId,
      scope: "locals"
    });
    assert.equal(varsC.locals.x, 4);

    // Continue Session B to completion
    const contB = await harness.callToolJson("shader_debug_continue", {
      sessionId: sessionB.sessionId
    });
    assert.equal(contB.status, "completed");
    assert.equal(contB.isAtEnd, true);

    // 5. Stop Session B first and verify Sessions A and C continue running normally
    const stopB = await harness.callToolJson("shader_debug_stop", {
      sessionId: sessionB.sessionId
    });
    assert.equal(stopB.disposed, true);
    assert.equal(harness.sessionManager.listSessions().length, 2);

    // Session A should still step and evaluate
    const evalA = await harness.callToolJson("shader_debug_eval", {
      sessionId: sessionA.sessionId,
      expression: "pos"
    });
    assert.equal(evalA.success, true);
    assert.deepEqual(evalA.value, [-0.5, -0.5]);

    // Complete and dispose remaining sessions
    await harness.callToolJson("shader_debug_stop", { sessionId: sessionA.sessionId });
    await harness.callToolJson("shader_debug_stop", { sessionId: sessionC.sessionId });

    assert.equal(harness.sessionManager.listSessions().length, 0);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 2. Full Debug Workflow on Real WoodWorks Capture: Vertex Debug Session on Draw #17
// ---------------------------------------------------------------------------
test("Full E2E debug workflow on real WoodWorks capture: vertex debug on draw #17 with captured vertex buffer", async () => {
  const capturePath = await findWoodWorksCapturePath();
  if (!capturePath) {
    console.log("    (Skipping real WoodWorks test: capture file not found)");
    return;
  }

  const harness = await createTestClient();
  try {
    // 1. Load real capture file via load_capture_file tool
    const loadRes = await harness.callToolJson("load_capture_file", {
      path: capturePath
    });
    assert(loadRes.captureId, "Must return valid captureId for WoodWorks");
    assert(loadRes.summary.totalObjects >= 100, "Capture should have at least 100 objects");
    const captureId = loadRes.captureId;

    // Verify draw #17 exists in capture commands
    const drawCmds = await harness.callToolJson("get_commands", {
      captureId,
      offset: 16,
      limit: 2
    });
    assert.equal(drawCmds.commands[0].method, "setVertexBuffer");
    assert.equal(drawCmds.commands[1].method, "draw");

    // 2. WGSL Vertex Shader matching pipeline #15 (sparse wood pipeline)
    const vertexShaderCode = `
      struct VertexInput {
        @location(0) position: vec3f,
      };

      struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) worldPos: vec3f,
        @location(1) scaleFactor: f32,
      };

      @vertex
      fn woodVertex(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        let scale = 1.5;
        let scaledPos = in.position * scale;
        out.pos = vec4f(scaledPos, 1.0);
        out.worldPos = scaledPos;
        out.scaleFactor = scale;
        return out;
      }
    `;

    // 3. Start vertex debug session on draw #17 for vertex index 1
    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId,
      commandIndex: 17,
      stage: "vertex",
      entryPoint: "woodVertex",
      code: vertexShaderCode,
      invocation: { vertexIndex: 1 }
    });

    assert(startRes.sessionId, "Must return active session ID");
    assert.equal(startRes.stage, "vertex");
    assert.equal(startRes.entryPoint, "woodVertex");
    assert.equal(startRes.status, "paused");
    assert(startRes.currentLine > 0);

    const sessionId = startRes.sessionId;

    // Verify vertex attribute @location(0) was decoded from the real captured vertex buffer #22
    // Vertex 1 in Wood proxy cube is [1.0, 0.0, 1.0]
    const inputVars = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      scope: "inputs"
    });
    assert(inputVars.inputs, "Inputs must be populated");
    assert.deepEqual(
      inputVars.inputs.position || inputVars.inputs["0"],
      [1.0, 0.0, 1.0],
      "Vertex #1 location 0 must match captured buffer data [1, 0, 1]"
    );

    // 4. Set breakpoint on line 18 (out.worldPos = scaledPos;)
    const bpRes = await harness.callToolJson("shader_debug_set_breakpoints", {
      sessionId,
      add: [18]
    });
    assert.deepEqual(bpRes.activeBreakpoints, [18]);

    // 5. Continue execution to hit breakpoint at line 18
    const contRes = await harness.callToolJson("shader_debug_continue", {
      sessionId
    });
    assert.equal(contRes.status, "paused");
    assert.equal(contRes.currentLine, 18);
    assert.equal(contRes.hitBreakpoint, 18);

    // 6. Evaluate vector attributes and local expressions at breakpoint
    // Evaluate input vector
    const evalInPos = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "in.position"
    });
    assert.equal(evalInPos.success, true);
    assert.deepEqual(evalInPos.value, [1, 0, 1]);

    // Evaluate vector component
    const evalInPosX = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "in.position.x"
    });
    assert.equal(evalInPosX.success, true);
    assert.equal(evalInPosX.value, 1.0);

    const evalInPosZ = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "in.position.z"
    });
    assert.equal(evalInPosZ.success, true);
    assert.equal(evalInPosZ.value, 1.0);

    // Evaluate vector swizzle
    const evalSwizzle = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "in.position.zx"
    });
    assert.equal(evalSwizzle.success, true);
    assert.deepEqual(evalSwizzle.value, [1.0, 1.0]);

    // Evaluate local variable scaledPos (1.0 * 1.5, 0.0 * 1.5, 1.0 * 1.5) = [1.5, 0, 1.5]
    const evalScaled = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "scaledPos"
    });
    assert.equal(evalScaled.success, true);
    assert.deepEqual(evalScaled.value, [1.5, 0.0, 1.5]);

    // Evaluate struct member out.pos (assigned on line 17)
    const evalOutPos = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "out.pos"
    });
    assert.equal(evalOutPos.success, true);
    assert.deepEqual(evalOutPos.value, [1.5, 0.0, 1.5, 1.0]);

    // 7. Step through remaining lines to completion
    const step1 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next"
    });
    assert.equal(step1.currentLine, 19); // out.scaleFactor = scale;

    const step2 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next"
    });
    assert.equal(step2.currentLine, 20); // return out;

    const step3 = await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next"
    });
    assert.equal(step3.status, "completed");
    assert.equal(step3.isAtEnd, true);
    assert(step3.returnValue, "Must have valid return value");
    assert.deepEqual(step3.returnValue.pos, [1.5, 0.0, 1.5, 1.0]);
    assert.deepEqual(step3.returnValue.worldPos, [1.5, 0.0, 1.5]);
    assert.equal(step3.returnValue.scaleFactor, 1.5);

    // 8. Clean stop and verify disposal
    const stopRes = await harness.callToolJson("shader_debug_stop", {
      sessionId
    });
    assert.equal(stopRes.sessionId, sessionId);
    assert.equal(stopRes.disposed, true);
    assert.equal(harness.sessionManager.getSession(sessionId), null);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 3. Full Debug Workflow on Real WoodWorks Capture: Compute Debug with Real Uniforms
// ---------------------------------------------------------------------------
test("Full E2E debug workflow on real WoodWorks capture: compute debug with real uniforms/storage buffers", async () => {
  const capturePath = await findWoodWorksCapturePath();
  if (!capturePath) {
    console.log("    (Skipping real WoodWorks compute test: capture file not found)");
    return;
  }

  const harness = await createTestClient();
  try {
    const loadRes = await harness.callToolJson("load_capture_file", {
      path: capturePath
    });
    const captureId = loadRes.captureId;

    // Compute Shader that binds Frame Uniforms (Buffer #20) and Output Storage Buffer
    const computeShaderCode = `
      struct FrameUniforms {
        viewProj: mat4x4f,
        invViewProj: mat4x4f,
        prevViewProj: mat4x4f,
        cameraPos: vec4f,
      };

      @group(0) @binding(0) var<uniform> uFrame: FrameUniforms;
      @group(0) @binding(1) var<storage, read_write> uOutput: array<vec4f>;

      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) gid: vec3u) {
        let camPos = uFrame.cameraPos;
        let camHeight = camPos.y;
        let pWorld = vec4f(f32(gid.x), camHeight, 0.0, 1.0);
        let pClip = uFrame.viewProj * pWorld;
        uOutput[gid.x] = pClip;
      }
    `;

    // 1. Start compute debug session at draw #17 (which binds Frame bind group at slot 0)
    const startRes = await harness.callToolJson("shader_debug_start", {
      captureId,
      commandIndex: 17,
      stage: "compute",
      entryPoint: "main",
      code: computeShaderCode,
      invocation: { threadId: [0, 0, 0] }
    });

    assert(startRes.sessionId, "Must create compute session on real capture");
    assert.equal(startRes.stage, "compute");
    assert.equal(startRes.status, "paused");

    const sessionId = startRes.sessionId;

    // 2. Inspect globals to verify real captured Frame uniforms
    const globalVars = await harness.callToolJson("shader_debug_get_variables", {
      sessionId,
      scope: "globals"
    });
    assert(globalVars.globals?.uFrame, "uFrame uniform struct must be present in globals");
    const frame = globalVars.globals.uFrame;

    // Check camera position matches capture Frame uniforms Buffer #20
    assert(Array.isArray(frame.cameraPos));
    assert.equal(frame.cameraPos.length, 4);
    assert.equal(Number(frame.cameraPos[0].toFixed(3)), -0.018);
    assert.equal(Number(frame.cameraPos[1].toFixed(3)), 0.143);
    assert.equal(Number(frame.cameraPos[2].toFixed(3)), 0.070);
    assert.equal(frame.cameraPos[3], 1.0);

    // 3. Step execution through the compute shader
    // Step to let camPos be computed
    await harness.callToolJson("shader_debug_step", {
      sessionId,
      action: "step_next",
      count: 2
    });

    // 4. Evaluate expressions
    // Evaluate uniform struct members
    const evalCamPos = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "uFrame.cameraPos"
    });
    assert.equal(evalCamPos.success, true);
    assert.equal(Number(evalCamPos.value[1].toFixed(3)), 0.143);

    // Evaluate swizzles on uniforms
    const evalCamSwizzle = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "uFrame.cameraPos.xyz"
    });
    assert.equal(evalCamSwizzle.success, true);
    assert.equal(evalCamSwizzle.value.length, 3);

    // Evaluate local variables
    const evalCamHeight = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "camHeight"
    });
    assert.equal(evalCamHeight.success, true);
    assert.equal(Number(evalCamHeight.value.toFixed(3)), 0.143);

    // Evaluate matrix column indexing
    const evalMatCol = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "uFrame.viewProj[0]"
    });
    assert.equal(evalMatCol.success, true);
    assert.equal(evalMatCol.value.length, 4);

    // 5. Continue to completion
    const contRes = await harness.callToolJson("shader_debug_continue", {
      sessionId
    });
    assert.equal(contRes.status, "completed");
    assert.equal(contRes.isAtEnd, true);

    // 6. Clean stop
    const stopRes = await harness.callToolJson("shader_debug_stop", {
      sessionId
    });
    assert.equal(stopRes.disposed, true);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 4. Multi-Invocation Debugging on the Same Real Capture
// ---------------------------------------------------------------------------
test("Multi-invocation debugging: parallel debug sessions on vertices #0 and #2 of real capture", async () => {
  const capturePath = await findWoodWorksCapturePath();
  if (!capturePath) {
    console.log("    (Skipping real WoodWorks multi-invocation test: capture file not found)");
    return;
  }

  const harness = await createTestClient();
  try {
    const loadRes = await harness.callToolJson("load_capture_file", { path: capturePath });
    const captureId = loadRes.captureId;

    const vCode = `
      struct In { @location(0) pos: vec3f };
      @vertex
      fn main(in: In) -> @builtin(position) vec4f {
        let p = in.pos;
        return vec4f(p, 1.0);
      }
    `;

    // Session 0: Vertex 0 (expected pos [0, 0, 1])
    const session0 = await harness.callToolJson("shader_debug_start", {
      captureId,
      commandIndex: 17,
      stage: "vertex",
      code: vCode,
      invocation: { vertexIndex: 0 }
    });

    // Session 2: Vertex 2 (expected pos [1, 1, 1])
    const session2 = await harness.callToolJson("shader_debug_start", {
      captureId,
      commandIndex: 17,
      stage: "vertex",
      code: vCode,
      invocation: { vertexIndex: 2 }
    });

    // Step both sessions
    await harness.callToolJson("shader_debug_step", { sessionId: session0.sessionId, count: 2 });
    await harness.callToolJson("shader_debug_step", { sessionId: session2.sessionId, count: 2 });

    // Evaluate pos in Session 0
    const eval0 = await harness.callToolJson("shader_debug_eval", {
      sessionId: session0.sessionId,
      expression: "p"
    });
    assert.deepEqual(eval0.value, [0, 0, 1]);

    // Evaluate pos in Session 2
    const eval2 = await harness.callToolJson("shader_debug_eval", {
      sessionId: session2.sessionId,
      expression: "p"
    });
    assert.deepEqual(eval2.value, [1, 1, 1]);

    // Cleanup both sessions
    await harness.callToolJson("shader_debug_stop", { sessionId: session0.sessionId });
    await harness.callToolJson("shader_debug_stop", { sessionId: session2.sessionId });

    assert.equal(harness.sessionManager.listSessions().length, 0);
  } finally {
    await harness.close();
  }
});

// ---------------------------------------------------------------------------
// 5. Error handling and resource isolation
// ---------------------------------------------------------------------------
test("Error handling: invalid expressions, expired sessions, and missing identifiers", async () => {
  const harness = await createTestClient();
  try {
    const sampleCap = createSampleCapture();
    const meta = await harness.loadSyntheticCapture(sampleCap);

    const session = await harness.callToolJson("shader_debug_start", {
      captureId: meta.id,
      commandIndex: 6,
      stage: "vertex",
      invocation: { vertexIndex: 0 }
    });
    const sessionId = session.sessionId;

    // Invalid evaluation syntax
    const badSyntax = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "pos...x"
    });
    assert.equal(badSyntax.success, false);
    assert(badSyntax.error);

    // Non-existent variable
    const missingVar = await harness.callToolJson("shader_debug_eval", {
      sessionId,
      expression: "unknownIdentifier"
    });
    assert.equal(missingVar.success, false);
    assert(missingVar.error.includes("not found"));

    // Stop session
    await harness.callToolJson("shader_debug_stop", { sessionId });

    // Calling tools with closed sessionId should throw error
    await assert.rejects(
      async () => {
        await harness.callToolJson("shader_debug_get_stack", { sessionId });
      },
      (err) => err.message.includes(`No active shader debug session "${sessionId}"`)
    );
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

  console.log(`\nRunning ${tests.length} E2E Shader Debugging Tests...\n`);

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
