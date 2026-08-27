#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  ShaderDebugSession,
  ShaderDebugSessionManager,
  formatDataValue,
  decodeTypedValue
} from "../src/shader-debug-session.js";

const tests = [];

function test(name, fn) {
  tests.push({ name, fn });
}

// ---------------------------------------------------------------------------
// 1. Compute kernel stepping: stepOver, stepInto, stepOut, stepNext
// ---------------------------------------------------------------------------
test("Compute kernel step_over skips helper function bodies", async () => {
  const code = `
fn helper(val: u32) -> u32 {
  let temp = val * 3u;
  return temp + 1u;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let a = 5u;
  let b = helper(a);
  let c = b + 10u;
}
`;

  const session = new ShaderDebugSession({
    code,
    stage: "compute",
    entryPoint: "main",
    invocation: { threadId: [0, 0, 0] }
  });

  const initSnap = session.init();
  assert.equal(initSnap.status, "paused");
  assert.equal(initSnap.currentFunction, "main");
  assert.equal(initSnap.currentLine, 9); // let a = 5u;

  // Step next to line 10
  const snap1 = session.step("step_next");
  assert.equal(snap1.currentLine, 10); // let b = helper(a);
  assert.equal(snap1.currentFunction, "main");

  // Step over helper function
  const snap2 = session.step("step_over");
  assert.equal(snap2.currentLine, 11); // let c = b + 10u;
  assert.equal(snap2.currentFunction, "main");

  // Check variables: b should now be computed (5 * 3 + 1 = 16)
  const vars = session.getVariables({ scope: "locals" });
  assert.equal(vars.locals.a, 5);
  assert.equal(vars.locals.b, 16);

  // Step to end
  const snap3 = session.step("step_next");
  assert.equal(snap3.status, "completed");
  assert.equal(snap3.isAtEnd, true);
});

test("Compute kernel step_into enters helper function and step_out returns to caller", async () => {
  const code = `
fn multiply(x: u32, y: u32) -> u32 {
  let prod = x * y;
  let extra = 2u;
  return prod + extra;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = 4u;
  let y = 6u;
  let res = multiply(x, y);
  let finalVal = res * 2u;
}
`;

  const session = new ShaderDebugSession({
    code,
    stage: "compute",
    entryPoint: "main"
  });

  session.init();
  assert.equal(session.debugger.currentLine, 10); // let x = 4u

  session.step("step_into"); // line 11 (let y = 6u)
  assert.equal(session.debugger.currentLine, 11);

  session.step("step_into"); // line 12 (let res = multiply(x, y))
  assert.equal(session.debugger.currentLine, 12);

  // Step into multiply
  session.step("step_into");
  assert.equal(session.debugger.currentLine, 3); // let prod = x * y
  const callstackInHelper = session.getCallstack();
  assert.equal(callstackInHelper.length, 2);
  assert.equal(callstackInHelper[0].functionName, "multiply");
  assert.equal(callstackInHelper[1].functionName, "main");

  // Step one line inside multiply
  session.step("step_next");
  assert.equal(session.debugger.currentLine, 4); // let extra = 2u

  // Step out back to main
  const snapOut = session.step("step_out");
  assert.equal(snapOut.currentFunction, "main");
  assert.equal(snapOut.currentLine, 12);

  // Next step in main
  session.step("step_next");
  assert.equal(session.debugger.currentLine, 13); // let finalVal = res * 2u
  const vars = session.getVariables({ scope: "locals" });
  assert.equal(vars.locals.res, 26); // 4 * 6 + 2 = 26
});

test("Multi-step execution with count parameter", async () => {
  const code = `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let a = 1u;
  let b = 2u;
  let c = 3u;
  let d = 4u;
}
`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  session.init();
  assert.equal(session.debugger.currentLine, 4);

  // Step 2 lines forward
  const snap = session.step("step_next", 2);
  assert.equal(snap.currentLine, 6);
  const vars = session.getVariables({ scope: "locals" });
  assert.equal(vars.locals.a, 1);
  assert.equal(vars.locals.b, 2);
});

// ---------------------------------------------------------------------------
// 2. Breakpoints add/remove/clear and continueExecution stopping at breakpoint
// ---------------------------------------------------------------------------
test("Breakpoints add, remove, clear and continueExecution", async () => {
  const code = `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  var acc = 0u;
  acc = acc + 10u;
  acc = acc + 20u;
  acc = acc + 30u;
  acc = acc + 40u;
}
`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  session.init();

  // Add breakpoint at line 6 (acc = acc + 20u) and line 8 (acc = acc + 40u)
  const bpList = session.setBreakpoints({ add: [6, 8] });
  assert.deepEqual(bpList, [6, 8]);
  assert.deepEqual(session.getStateSnapshot().activeBreakpoints, [6, 8]);

  // continueExecution should hit line 6
  const snap1 = session.continueExecution();
  assert.equal(snap1.status, "paused");
  assert.equal(snap1.currentLine, 6);
  assert.equal(snap1.hitBreakpoint, 6);
  assert.equal(session.getVariables({ scope: "locals" }).locals.acc, 10);

  // continueExecution should hit line 8
  const snap2 = session.continueExecution();
  assert.equal(snap2.status, "paused");
  assert.equal(snap2.currentLine, 8);
  assert.equal(snap2.hitBreakpoint, 8);
  assert.equal(session.getVariables({ scope: "locals" }).locals.acc, 60);

  // Remove line 8 and continue to end
  session.setBreakpoints({ remove: [8] });
  assert.deepEqual(session.getStateSnapshot().activeBreakpoints, [6]);

  const snap3 = session.continueExecution();
  assert.equal(snap3.status, "completed");
  assert.equal(snap3.isAtEnd, true);
  assert.equal(session.getVariables({ scope: "locals" }).locals.acc, 100);

  // Clear all breakpoints
  session.setBreakpoints({ clearAll: true });
  assert.deepEqual(session.getStateSnapshot().activeBreakpoints, []);
});

// ---------------------------------------------------------------------------
// 3. Callstack unwinding across helper functions and block scopes
// ---------------------------------------------------------------------------
test("Callstack unwinding across nested helper functions and block scopes", async () => {
  const code = `
fn level3(v: u32) -> u32 {
  if (v > 0u) {
    let bonus = 100u;
    return v + bonus;
  }
  return v;
}

fn level2(val: u32) -> u32 {
  var sum = 0u;
  for (var i = 0u; i < 1u; i = i + 1u) {
    sum = sum + level3(val + i);
  }
  return sum;
}

fn level1(x: u32) -> u32 {
  let y = level2(x * 2u);
  return y;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let start = 5u;
  let result = level1(start);
}
`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  session.init();

  // Set breakpoint inside level3 at line 5 (let bonus = 100u)
  session.setBreakpoints({ add: [5] });

  const snap = session.continueExecution();
  assert.equal(snap.status, "paused");
  assert.equal(snap.currentLine, 5);
  assert.equal(snap.currentFunction, "level3");

  // Verify callstack is collapsed properly (innermost first)
  const stack = session.getCallstack();
  assert.equal(stack.length, 4, `Expected callstack depth 4, got ${stack.length}`);

  assert.equal(stack[0].level, 0);
  assert.equal(stack[0].functionName, "level3");
  assert.equal(stack[0].line, 5);

  assert.equal(stack[1].level, 1);
  assert.equal(stack[1].functionName, "level2");
  assert.equal(stack[1].line, 13); // line calling level3

  assert.equal(stack[2].level, 2);
  assert.equal(stack[2].functionName, "level1");
  assert.equal(stack[2].line, 19); // line calling level2

  assert.equal(stack[3].level, 3);
  assert.equal(stack[3].functionName, "main");
  assert.equal(stack[3].line, 26); // line calling level1
});

// ---------------------------------------------------------------------------
// 4. Local and global variable inspection and formatting
// ---------------------------------------------------------------------------
test("Variable inspection across locals, inputs, globals, constants, and resources", async () => {
  const code = `
struct Config {
  scale: f32,
  offset: vec2f,
  flag: u32,
};

@group(0) @binding(0) var<uniform> uConfig: Config;
@group(0) @binding(1) var<storage, read_write> sOutput: array<f32>;
@group(0) @binding(2) var tMask: texture_2d<f32>;
@group(0) @binding(3) var sSampler: sampler;

override OVERRIDE_VAL: u32 = 77u;
const MY_CONST: f32 = 2.718;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let localNum = 42u;
  let localVec = vec3f(1.0, 2.0, 3.0);
  let localMat = mat2x2f(1.0, 2.0, 3.0, 4.0);
  let localArr = array<u32, 3>(10u, 20u, 30u);
  let localConfig = Config(uConfig.scale * 2.0, uConfig.offset, 1u);
}
`;

  // Construct uniform buffer
  const uniformBuf = new ArrayBuffer(32);
  const f32Uniform = new Float32Array(uniformBuf);
  const u32Uniform = new Uint32Array(uniformBuf);
  f32Uniform[0] = 3.5; // scale
  f32Uniform[2] = 0.5; f32Uniform[3] = 1.5; // offset (vec2f align 8)
  u32Uniform[4] = 1; // flag

  const storageBuf = new Float32Array([100.0, 200.0]).buffer;

  const bindGroups = {
    0: {
      0: { uniform: uniformBuf },
      1: storageBuf,
      2: { texture: {}, descriptor: { size: [64, 64, 1] } },
      3: { sampler: { minFilter: "linear" } }
    }
  };

  const session = new ShaderDebugSession({
    code,
    stage: "compute",
    bindGroups,
    options: { constants: { OVERRIDE_VAL: 77 } }
  });

  session.init();
  session.continueExecution();

  const allVars = session.getVariables({ scope: "all" });

  // 1. Locals
  assert(allVars.locals, "Locals should exist");
  assert.equal(allVars.locals.localNum, 42);
  assert.deepEqual(allVars.locals.localVec, [1, 2, 3]);
  assert.deepEqual(allVars.locals.localMat, [1, 2, 3, 4]);
  assert.deepEqual(allVars.locals.localArr, [10, 20, 30]);
  assert.equal(allVars.locals.localConfig.scale, 7.0);
  assert.deepEqual(allVars.locals.localConfig.offset, [0.5, 1.5]);
  assert.equal(allVars.locals.localConfig.flag, 1);

  // 2. Globals (Uniforms & Storage)
  assert(allVars.globals.uConfig, "uConfig uniform should exist");
  assert.equal(allVars.globals.uConfig.scale, 3.5);
  assert.deepEqual(allVars.globals.uConfig.offset, [0.5, 1.5]);
  assert.equal(allVars.globals.uConfig.flag, 1);

  // 3. Inputs
  assert(allVars.inputs["@global_invocation_id"] || allVars.inputs.gid);

  // 4. Constants
  assert.equal(allVars.constants.OVERRIDE_VAL, 77);

  // 5. Resources
  assert(allVars.resources.tMask, "Texture should exist in resources");
  assert(allVars.resources.sSampler, "Sampler should exist in resources");

  // Filtering
  const filtered = session.getVariables({ filter: "localV" });
  assert(filtered.locals.localVec, "Filtered locals should contain localVec");
  assert.equal(filtered.locals.localNum, undefined, "Filtered locals should omit localNum");
});

// ---------------------------------------------------------------------------
// 5. Variable path evaluation (`foo.bar[0]`, swizzles, matrices)
// ---------------------------------------------------------------------------
test("Variable path evaluation supporting structs, arrays, vectors, and swizzling", async () => {
  const code = `
struct Point {
  pos: vec3f,
  weight: f32,
};

struct Mesh {
  name_id: u32,
  point0: Point,
  point1: Point,
  normal: vec4f,
};

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p0 = Point(vec3f(1.0, 2.0, 3.0), 0.5);
  let p1 = Point(vec3f(4.0, 5.0, 6.0), 0.75);
  let mesh = Mesh(99u, p0, p1, vec4f(0.1, 0.2, 0.3, 1.0));
  var points = array<Point, 2>(p0, p1);
  let testVec = vec4f(10.0, 20.0, 30.0, 40.0);
}
`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  session.init();
  session.continueExecution();

  // Direct scalar evaluation
  const evalId = session.evaluate("mesh.name_id");
  assert.equal(evalId.success, true);
  assert.equal(evalId.value, 99);

  // Nested struct member access
  const evalP0 = session.evaluate("mesh.point0.pos");
  assert.equal(evalP0.success, true);
  assert.deepEqual(evalP0.value, [1, 2, 3]);

  const evalP0Weight = session.evaluate("mesh.point0.weight");
  assert.equal(evalP0Weight.success, true);
  assert.equal(evalP0Weight.value, 0.5);

  const evalP1X = session.evaluate("mesh.point1.pos.x");
  assert.equal(evalP1X.success, true);
  assert.equal(evalP1X.value, 4);

  // Array of structs indexing
  const evalArrP0 = session.evaluate("points[0].pos");
  assert.equal(evalArrP0.success, true);
  assert.deepEqual(evalArrP0.value, [1, 2, 3]);

  const evalArrP1Weight = session.evaluate("points[1].weight");
  assert.equal(evalArrP1Weight.success, true);
  assert.equal(evalArrP1Weight.value, 0.75);

  // Swizzles on vector
  const evalSwizzle1 = session.evaluate("testVec.x");
  assert.equal(evalSwizzle1.success, true);
  assert.equal(evalSwizzle1.value, 10);

  const evalSwizzle2 = session.evaluate("testVec.xyz");
  assert.equal(evalSwizzle2.success, true);
  assert.deepEqual(evalSwizzle2.value, [10, 20, 30]);

  const evalSwizzle3 = session.evaluate("mesh.normal.rgb");
  assert.equal(evalSwizzle3.success, true);
  assert.deepEqual(
    evalSwizzle3.value.map((v) => Number(v.toFixed(2))),
    [0.1, 0.2, 0.3]
  );

  // Vector index access
  const evalVecIdx = session.evaluate("testVec[2]");
  assert.equal(evalVecIdx.success, true);
  assert.equal(evalVecIdx.value, 30);

  // Error cases
  const evalMissing = session.evaluate("mesh.nonExistent");
  assert.equal(evalMissing.success, false);

  const evalOutOfBounds = session.evaluate("points[99]");
  assert.equal(evalOutOfBounds.success, false);

  const evalBadSyntax = session.evaluate("mesh...point0");
  assert.equal(evalBadSyntax.success, false);
});

// ---------------------------------------------------------------------------
// 6. Vertex & Fragment shader debugging
// ---------------------------------------------------------------------------
test("Vertex and Fragment shader debugging sessions", async () => {
  const vertCode = `
struct VertexInput {
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) position: vec2f,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.pos = vec4f(in.position, 0.0, 1.0);
  out.uv = vec2f(f32(in.vertexIndex) * 0.5, 1.0);
  return out;
}
`;

  const vertSession = new ShaderDebugSession({
    code: vertCode,
    stage: "vertex",
    entryPoint: "vs_main",
    invocation: {
      vertexIndex: 2,
      position: [0.75, -0.25]
    }
  });

  const vSnap = vertSession.init();
  assert.equal(vSnap.status, "paused");
  vertSession.continueExecution();
  assert.equal(vertSession.isAtEnd, true);

  const vertRet = vertSession.debugger.getReturnValue();
  assert(vertRet, "Vertex shader should have return value");
  assert.deepEqual(vertRet.pos, [0.75, -0.25, 0, 1]);
  assert.deepEqual(vertRet.uv, [1, 1]);

  // Fragment Shader
  const fragCode = `
@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let col = vec4f(uv.x, uv.y, 0.5, 1.0);
  return col;
}
`;

  const fragSession = new ShaderDebugSession({
    code: fragCode,
    stage: "fragment",
    entryPoint: "fs_main",
    invocation: {
      0: [0.3, 0.7]
    }
  });

  fragSession.init();
  fragSession.continueExecution();
  assert.equal(fragSession.isAtEnd, true);

  const fragRet = fragSession.debugger.getReturnValue();
  assert(fragRet, "Fragment shader should return color");
  assert.deepEqual(
    Array.from(fragRet).map((v) => Number(v.toFixed(1))),
    [0.3, 0.7, 0.5, 1.0]
  );
});

// ---------------------------------------------------------------------------
// 7. Source Snippet generation
// ---------------------------------------------------------------------------
test("Source snippet generation formats line numbers and arrow marker", async () => {
  const code = `fn computeVal(x: u32) -> u32 {
  let a = x + 1u;
  let b = a * 2u;
  return b;
}`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  const snippet = session.getSourceSnippet(3, 1);
  assert(snippet.includes("-> 3 |   let b = a * 2u;"), `Expected marker on line 3, got:\n${snippet}`);
  assert(snippet.includes("   2 |   let a = x + 1u;"), `Expected context line 2, got:\n${snippet}`);
  assert(snippet.includes("   4 |   return b;"), `Expected context line 4, got:\n${snippet}`);
});

// ---------------------------------------------------------------------------
// 8. ShaderDebugSessionManager: creation, get, list, TTL cleanup, and delete
// ---------------------------------------------------------------------------
test("ShaderDebugSessionManager handles session lifecycle and TTL cleanup", async () => {
  const manager = new ShaderDebugSessionManager();

  const code = `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let v = 100u;
}
`;

  // 1. Create Session
  const session1 = manager.createSession({ code, stage: "compute" });
  assert(session1, "Session should be created");
  assert.equal(typeof session1.id, "string");
  assert.equal(session1.status, "paused");

  const session2 = manager.createSession({ code, stage: "compute" });

  // 2. List Sessions
  const list = manager.listSessions();
  assert.equal(list.length, 2);
  const ids = list.map((s) => s.sessionId);
  assert(ids.includes(session1.id));
  assert(ids.includes(session2.id));

  // 3. Get Session
  const retrieved = manager.getSession(session1.id);
  assert.equal(retrieved.id, session1.id);

  // 4. Delete Session
  const deleted = manager.deleteSession(session1.id);
  assert.equal(deleted, true);
  assert.equal(manager.getSession(session1.id), null);
  assert.equal(session1.status, "disposed");
  assert.equal(manager.listSessions().length, 1);

  // 5. TTL Expiration Cleanup
  // Simulate session2 being old
  session2.lastAccessedAt = Date.now() - 20 * 60 * 1000; // 20 minutes ago
  const expiredCount = manager.cleanupExpired(15 * 60 * 1000);
  assert.equal(expiredCount, 1);
  assert.equal(manager.listSessions().length, 0);

  // 6. getSession on expired session
  const session3 = manager.createSession({ code, stage: "compute" });
  session3.lastAccessedAt = Date.now() - 30 * 60 * 1000;
  const expiredGet = manager.getSession(session3.id, 15 * 60 * 1000);
  assert.equal(expiredGet, null);
  assert.equal(session3.status, "disposed");
});

// ---------------------------------------------------------------------------
// 9. Edge Cases: Matrix Indexing, Swizzle Bounds, Chained Swizzles, and Scope Isolation
// ---------------------------------------------------------------------------
test("Edge cases: Matrix column indexing, swizzle bounds validation, and scope isolation", async () => {
  const code = `
fn calc(a: f32) -> f32 {
  let localInHelper = a * 10.0;
  return localInHelper;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let mat3 = mat3x3f(
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
  );
  let v2 = vec2f(10.0, 20.0);
  let v4 = vec4f(1.0, 2.0, 3.0, 4.0);
  let res = calc(v2.x);
}
`;

  const session = new ShaderDebugSession({ code, stage: "compute" });
  session.init();

  // Set breakpoint inside calc at line 3 (return localInHelper;)
  session.setBreakpoints({ add: [4] });
  session.continueExecution();

  assert.equal(session.getStateSnapshot().currentFunction, "calc");
  const helperVars = session.getVariables({ scope: "locals" });
  assert.equal(helperVars.locals.localInHelper, 100);
  // main's locals should not be in helper's locals
  assert.equal(helperVars.locals.mat3, undefined);
  assert.equal(helperVars.locals.v2, undefined);

  // Step out back to main and complete
  session.step("step_out");
  session.step("step_next");

  // Matrix indexing
  const evalCol1 = session.evaluate("mat3[1]");
  assert.equal(evalCol1.success, true);
  assert.deepEqual(evalCol1.value, [4, 5, 6]);

  const evalCol1Row2 = session.evaluate("mat3[1][2]");
  assert.equal(evalCol1Row2.success, true);
  assert.equal(evalCol1Row2.value, 6);

  // Swizzle bounds validation
  const evalBadSwizzle = session.evaluate("v2.z");
  assert.equal(evalBadSwizzle.success, false);
  assert(evalBadSwizzle.error.includes("out of bounds"));

  // Chained swizzles
  const evalChained = session.evaluate("v4.xyz.xy");
  assert.equal(evalChained.success, true);
  assert.deepEqual(evalChained.value, [1, 2]);
});

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------
async function runAll() {
  let passed = 0;
  let failed = 0;
  const startTime = Date.now();

  console.log(`\nRunning ${tests.length} Shader Debug Session Unit Tests...\n`);

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
