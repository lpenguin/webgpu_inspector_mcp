#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  VERTEX_FORMATS,
  decodeVertexAttribute,
  fetchVertexInputs,
  buildSessionBindGroups,
  prepareFragmentInputs,
  buildFragmentQuadInputs,
  prepareShaderDebugSession,
  toArrayBuffer,
  resolvePassState
} from "../src/stage-adapters.js";
import { ShaderDebugSession } from "../src/shader-debug-session.js";

const tests = [];

function test(name, fn) {
  tests.push({ name, fn });
}

// ---------------------------------------------------------------------------
// 1. Vertex Attribute Decoding across all standard WebGPU formats
// ---------------------------------------------------------------------------
test("Vertex attribute decoding for 8-bit formats (uint8, sint8, unorm8, snorm8)", async () => {
  const buf = new ArrayBuffer(16);
  const view = new DataView(buf);

  // uint8 / uint8x2 / uint8x4
  view.setUint8(0, 10);
  view.setUint8(1, 20);
  view.setUint8(2, 30);
  view.setUint8(3, 40);
  assert.equal(decodeVertexAttribute(view, 0, "uint8"), 10);
  assert.deepEqual(decodeVertexAttribute(view, 0, "uint8x2"), [10, 20]);
  assert.deepEqual(decodeVertexAttribute(view, 0, "uint8x4"), [10, 20, 30, 40]);

  // sint8 / sint8x2 / sint8x4
  view.setInt8(4, -10);
  view.setInt8(5, 20);
  view.setInt8(6, -30);
  view.setInt8(7, 40);
  assert.equal(decodeVertexAttribute(view, 4, "sint8"), -10);
  assert.deepEqual(decodeVertexAttribute(view, 4, "sint8x2"), [-10, 20]);
  assert.deepEqual(decodeVertexAttribute(view, 4, "sint8x4"), [-10, 20, -30, 40]);

  // unorm8 / unorm8x2 / unorm8x4
  view.setUint8(8, 0);
  view.setUint8(9, 255);
  view.setUint8(10, 128);
  view.setUint8(11, 64);
  assert.equal(decodeVertexAttribute(view, 8, "unorm8"), 0.0);
  assert.deepEqual(decodeVertexAttribute(view, 8, "unorm8x2"), [0.0, 1.0]);
  const unorm4 = decodeVertexAttribute(view, 8, "unorm8x4");
  assert.equal(unorm4[0], 0.0);
  assert.equal(unorm4[1], 1.0);
  assert.equal(Number(unorm4[2].toFixed(3)), Number((128 / 255).toFixed(3)));

  // snorm8 / snorm8x2 / snorm8x4
  view.setInt8(12, 0);
  view.setInt8(13, 127);
  view.setInt8(14, -128);
  view.setInt8(15, -64);
  assert.equal(decodeVertexAttribute(view, 12, "snorm8"), 0.0);
  const snorm4 = decodeVertexAttribute(view, 12, "snorm8x4");
  assert.equal(snorm4[0], 0.0);
  assert.equal(snorm4[1], 1.0);
  assert.equal(snorm4[2], -1.0);
  assert.equal(Number(snorm4[3].toFixed(3)), Number((-64 / 127).toFixed(3)));
});

test("Vertex attribute decoding for 16-bit formats (uint16, sint16, unorm16, snorm16, float16)", async () => {
  const buf = new ArrayBuffer(32);
  const view = new DataView(buf);

  // uint16 / uint16x2 / uint16x4
  view.setUint16(0, 1000, true);
  view.setUint16(2, 2000, true);
  view.setUint16(4, 3000, true);
  view.setUint16(6, 4000, true);
  assert.equal(decodeVertexAttribute(view, 0, "uint16"), 1000);
  assert.deepEqual(decodeVertexAttribute(view, 0, "uint16x2"), [1000, 2000]);
  assert.deepEqual(decodeVertexAttribute(view, 0, "uint16x4"), [1000, 2000, 3000, 4000]);

  // sint16 / sint16x2 / sint16x4
  view.setInt16(8, -1000, true);
  view.setInt16(10, 2000, true);
  assert.equal(decodeVertexAttribute(view, 8, "sint16"), -1000);
  assert.deepEqual(decodeVertexAttribute(view, 8, "sint16x2"), [-1000, 2000]);

  // unorm16 / snorm16
  view.setUint16(12, 65535, true);
  view.setInt16(14, 32767, true);
  assert.equal(decodeVertexAttribute(view, 12, "unorm16"), 1.0);
  assert.equal(decodeVertexAttribute(view, 14, "snorm16"), 1.0);

  // float16 (half-precision float: 0x3c00 = 1.0, 0xc000 = -2.0, 0x4200 = 3.0, 0x4400 = 4.0)
  view.setUint16(16, 0x3c00, true);
  view.setUint16(18, 0xc000, true);
  view.setUint16(20, 0x4200, true);
  view.setUint16(22, 0x4400, true);
  assert.equal(decodeVertexAttribute(view, 16, "float16"), 1.0);
  assert.deepEqual(decodeVertexAttribute(view, 16, "float16x2"), [1.0, -2.0]);
  assert.deepEqual(decodeVertexAttribute(view, 16, "float16x4"), [1.0, -2.0, 3.0, 4.0]);
});

test("Vertex attribute decoding for 32-bit formats (float32, uint32, sint32) and unorm10-10-10-2", async () => {
  const buf = new ArrayBuffer(64);
  const view = new DataView(buf);

  // float32x1..4
  view.setFloat32(0, 1.5, true);
  view.setFloat32(4, -2.5, true);
  view.setFloat32(8, 3.5, true);
  view.setFloat32(12, -4.5, true);
  assert.equal(decodeVertexAttribute(view, 0, "float32"), 1.5);
  assert.deepEqual(decodeVertexAttribute(view, 0, "float32x2"), [1.5, -2.5]);
  assert.deepEqual(decodeVertexAttribute(view, 0, "float32x3"), [1.5, -2.5, 3.5]);
  assert.deepEqual(decodeVertexAttribute(view, 0, "float32x4"), [1.5, -2.5, 3.5, -4.5]);

  // uint32 / sint32
  view.setUint32(16, 50000, true);
  view.setInt32(20, -50000, true);
  assert.equal(decodeVertexAttribute(view, 16, "uint32"), 50000);
  assert.equal(decodeVertexAttribute(view, 20, "sint32"), -50000);

  // unorm10-10-10-2 (packed 10-bit R, 10-bit G, 10-bit B, 2-bit A)
  // r=1023 (1.0), g=0 (0.0), b=511 (~0.5), a=3 (1.0)
  const u10 = 1023 | (0 << 10) | (511 << 20) | (3 << 30);
  view.setUint32(24, u10, true);
  const u10Decoded = decodeVertexAttribute(view, 24, "unorm10-10-10-2");
  assert.equal(u10Decoded[0], 1.0);
  assert.equal(u10Decoded[1], 0.0);
  assert.equal(Number(u10Decoded[2].toFixed(3)), Number((511 / 1023).toFixed(3)));
  assert.equal(u10Decoded[3], 1.0);
});

test("fetchVertexInputs supports stepMode: vertex and stepMode: instance across multiple slots", async () => {
  // Slot 0: Vertex buffer (stride 8: float32x2)
  const vb0Floats = new Float32Array([
    0.0, 0.5,   // Vertex 0
    -0.5, -0.5, // Vertex 1
    0.5, -0.5   // Vertex 2
  ]);

  // Slot 1: Instance buffer (stride 16: float32x4)
  const vb1Floats = new Float32Array([
    1.0, 0.0, 0.0, 1.0, // Instance 0 (Red)
    0.0, 1.0, 0.0, 1.0  // Instance 1 (Green)
  ]);

  const pipelineDesc = {
    vertex: {
      buffers: [
        {
          arrayStride: 8,
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }]
        },
        {
          arrayStride: 16,
          stepMode: "instance",
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x4" }]
        }
      ]
    }
  };

  const vertexBuffers = [
    { slot: 0, buffer: vb0Floats.buffer },
    { slot: 1, buffer: vb1Floats.buffer }
  ];

  // Vertex 1, Instance 1
  const inputs = fetchVertexInputs(pipelineDesc, vertexBuffers, 1, 1);
  assert.equal(inputs.vertex_index, 1);
  assert.equal(inputs.instance_index, 1);
  assert.deepEqual(inputs[0], [-0.5, -0.5]);
  assert.deepEqual(inputs[1], [0.0, 1.0, 0.0, 1.0]);

  // Vertex 0, Instance 0
  const inputs0 = fetchVertexInputs(pipelineDesc, vertexBuffers, 0, 0);
  assert.deepEqual(inputs0[0], [0.0, 0.5]);
  assert.deepEqual(inputs0[1], [1.0, 0.0, 0.0, 1.0]);
});

// ---------------------------------------------------------------------------
// 2. Compute dispatch with uniform & storage buffers via prepareShaderDebugSession
// ---------------------------------------------------------------------------
test("prepareShaderDebugSession on compute dispatch with uniform and storage buffers", async () => {
  const code = `
    struct Params {
      multiplier: f32,
      baseOffset: vec2f,
      mode: u32,
    };

    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage, read_write> outBuffer: array<vec4f>;

    @compute @workgroup_size(1)
    fn comp_main(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      let scaledX = (f32(idx) + params.baseOffset.x) * params.multiplier;
      let scaledY = (f32(idx) + params.baseOffset.y) * params.multiplier;
      outBuffer[idx] = vec4f(scaledX, scaledY, f32(params.mode), 1.0);
    }
  `;

  // Construct uniform buffer
  const uniformBuffer = new ArrayBuffer(32);
  const f32U = new Float32Array(uniformBuffer);
  const u32U = new Uint32Array(uniformBuffer);
  f32U[0] = 2.5; // multiplier
  f32U[2] = 10.0; f32U[3] = 20.0; // baseOffset (vec2f aligned to 8)
  u32U[4] = 99; // mode

  // Storage buffer (4 vec4fs = 64 bytes)
  const storageBuffer = new ArrayBuffer(64);

  const capture = {
    objects: {
      "1": {
        id: 1,
        type: "ShaderModule",
        descriptor: { code }
      },
      "2": {
        id: 2,
        type: "ComputePipeline",
        descriptor: {
          compute: { module: { __id: 1 }, entryPoint: "comp_main" }
        }
      },
      "3": {
        id: 3,
        type: "Buffer",
        size: 32,
        initialData: uniformBuffer
      },
      "4": {
        id: 4,
        type: "Buffer",
        size: 64,
        initialData: storageBuffer
      },
      "5": {
        id: 5,
        type: "BindGroup",
        descriptor: {
          entries: [
            { binding: 0, resource: { buffer: { __id: 3 } } },
            { binding: 1, resource: { buffer: { __id: 4 } } }
          ]
        }
      }
    },
    commands: [
      { method: "beginComputePass", object: "GPUCommandEncoder#1", args: [{ label: "compute_pass" }], result: "GPUComputePassEncoder#1" },
      { method: "setPipeline", object: "GPUComputePassEncoder#1", args: [{ __id: 2 }] },
      { method: "setBindGroup", object: "GPUComputePassEncoder#1", args: [0, { __id: 5 }] },
      { method: "dispatchWorkgroups", object: "GPUComputePassEncoder#1", args: [4, 1, 1] },
      { method: "end", object: "GPUComputePassEncoder#1", args: [] }
    ]
  };

  // Launch session for thread (2, 0, 0)
  const session = prepareShaderDebugSession({
    capture,
    commandIndex: 3,
    stage: "compute",
    invocation: { threadId: [2, 0, 0] }
  });

  assert(session instanceof ShaderDebugSession);
  assert.equal(session.stage, "compute");
  assert.equal(session.entryPoint, "comp_main");
  assert.equal(session.status, "paused");

  // Verify initial global variables
  const initialVars = session.getVariables({ scope: "globals" });
  assert.equal(initialVars.globals.params.multiplier, 2.5);
  assert.deepEqual(initialVars.globals.params.baseOffset, [10.0, 20.0]);
  assert.equal(initialVars.globals.params.mode, 99);

  // Run shader to completion
  const snap = session.continueExecution();
  assert.equal(snap.status, "completed");
  assert.equal(session.isAtEnd, true);

  // Verify local values
  const locals = session.getVariables({ scope: "locals" }).locals;
  assert.equal(locals.idx, 2);
  assert.equal(locals.scaledX, (2 + 10) * 2.5); // 30
  assert.equal(locals.scaledY, (2 + 20) * 2.5); // 55

  // Verify evaluate
  const evalX = session.evaluate("scaledX");
  assert.equal(evalX.success, true);
  assert.equal(evalX.value, 30);

  // Verify storage buffer memory was updated
  const storageOut = new Float32Array(storageBuffer);
  // index 2 -> offset 2 * 4 = 8
  assert.equal(storageOut[8], 30);
  assert.equal(storageOut[9], 55);
  assert.equal(storageOut[10], 99);
  assert.equal(storageOut[11], 1.0);
});

// ---------------------------------------------------------------------------
// 3. Vertex draw with vertex attributes and instance attributes via prepareShaderDebugSession
// ---------------------------------------------------------------------------
test("prepareShaderDebugSession on vertex draw with vertex attributes and instance attributes", async () => {
  const code = `
    struct VertexIn {
      @builtin(vertex_index) vIdx: u32,
      @builtin(instance_index) iIdx: u32,
      @location(0) position: vec2f,
      @location(1) uv: vec2f,
      @location(2) instColor: vec4f,
    };

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
      @location(1) texCoord: vec2f,
    };

    @vertex
    fn vs_main(in: VertexIn) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(in.position, 0.0, 1.0);
      out.color = in.instColor;
      out.texCoord = in.uv;
      return out;
    }
  `;

  // Slot 0: vertex buffer (stride 16: float32x2 pos + float32x2 uv)
  const vb0Floats = new Float32Array([
    0.0, 0.5,    0.5, 1.0,  // Vertex 0
    -0.5, -0.5,  0.0, 0.0,  // Vertex 1
    0.5, -0.5,   1.0, 0.0   // Vertex 2
  ]);

  // Slot 1: instance buffer (stride 16: float32x4 color)
  const vb1Floats = new Float32Array([
    1.0, 0.0, 0.0, 1.0, // Instance 0: Red
    0.0, 1.0, 0.0, 1.0  // Instance 1: Green
  ]);

  const capture = {
    objects: {
      "1": {
        id: 1,
        type: "ShaderModule",
        descriptor: { code }
      },
      "2": {
        id: 2,
        type: "RenderPipeline",
        descriptor: {
          vertex: {
            module: { __id: 1 },
            entryPoint: "vs_main",
            buffers: [
              {
                arrayStride: 16,
                stepMode: "vertex",
                attributes: [
                  { shaderLocation: 0, offset: 0, format: "float32x2" },
                  { shaderLocation: 1, offset: 8, format: "float32x2" }
                ]
              },
              {
                arrayStride: 16,
                stepMode: "instance",
                attributes: [
                  { shaderLocation: 2, offset: 0, format: "float32x4" }
                ]
              }
            ]
          }
        }
      },
      "3": {
        id: 3,
        type: "Buffer",
        size: 48,
        initialData: vb0Floats.buffer
      },
      "4": {
        id: 4,
        type: "Buffer",
        size: 32,
        initialData: vb1Floats.buffer
      }
    },
    commands: [
      { method: "beginRenderPass", object: "GPUCommandEncoder#1", args: [{ label: "render_pass" }], result: "GPURenderPassEncoder#1" },
      { method: "setPipeline", object: "GPURenderPassEncoder#1", args: [{ __id: 2 }] },
      { method: "setVertexBuffer", object: "GPURenderPassEncoder#1", args: [0, { __id: 3 }] },
      { method: "setVertexBuffer", object: "GPURenderPassEncoder#1", args: [1, { __id: 4 }] },
      { method: "draw", object: "GPURenderPassEncoder#1", args: [3, 2, 0, 0] },
      { method: "end", object: "GPURenderPassEncoder#1", args: [] }
    ]
  };

  // Test Vertex 1, Instance 1
  const session = prepareShaderDebugSession({
    capture,
    commandIndex: 4,
    stage: "vertex",
    invocation: { vertexIndex: 1, instanceIndex: 1 }
  });

  assert(session instanceof ShaderDebugSession);
  assert.equal(session.stage, "vertex");
  assert.equal(session.entryPoint, "vs_main");

  // Run to completion
  session.continueExecution();
  assert.equal(session.isAtEnd, true);

  const ret = session.debugger.getReturnValue();
  assert(ret, "Vertex shader should have return value");
  assert.deepEqual(ret.pos, [-0.5, -0.5, 0, 1]);
  assert.deepEqual(ret.color, [0.0, 1.0, 0.0, 1.0]);
  assert.deepEqual(ret.texCoord, [0.0, 0.0]);
});

// ---------------------------------------------------------------------------
// 4. Fragment draw with pixel coordinates and texture sampling via prepareShaderDebugSession
// ---------------------------------------------------------------------------
test("prepareShaderDebugSession on fragment draw with pixel coordinates and texture sampling", async () => {
  const vertCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };
    @vertex
    fn vs_main(@location(0) position: vec2f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(position, 0.0, 1.0);
      out.uv = uv;
      return out;
    }
  `;

  const fragCode = `
    @group(0) @binding(0) var tTex: texture_2d<f32>;
    @group(0) @binding(1) var sSamp: sampler;

    @fragment
    fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      let sampled = textureSample(tTex, sSamp, uv);
      return sampled;
    }
  `;

  // Screen-filling triangle (top-left at (0,0), bottom-right at (800, 600))
  // Vertex 0: (-1.0, 1.0) -> screen (0, 0), uv = (0, 0)
  // Vertex 1: (-1.0, -3.0) -> screen (0, 1200), uv = (0, 2)
  // Vertex 2: (3.0, 1.0) -> screen (1600, 0), uv = (2, 0)
  const vbFloats = new Float32Array([
    -1.0, 1.0,  0.0, 0.0,
    -1.0, -3.0, 0.0, 2.0,
    3.0, 1.0,   2.0, 0.0
  ]);

  // 2x2 Texture: (0,0)=Red [255,0,0,255], (1,0)=Green [0,255,0,255], (0,1)=Blue [0,0,255,255], (1,1)=White [255,255,255,255]
  const texPixels = new Uint8Array([
    255, 0, 0, 255,    0, 255, 0, 255,
    0, 0, 255, 255,    255, 255, 255, 255
  ]);

  const capture = {
    objects: {
      "1": { id: 1, type: "ShaderModule", descriptor: { code: vertCode } },
      "2": { id: 2, type: "ShaderModule", descriptor: { code: fragCode } },
      "3": {
        id: 3,
        type: "RenderPipeline",
        descriptor: {
          vertex: {
            module: { __id: 1 },
            entryPoint: "vs_main",
            buffers: [
              {
                arrayStride: 16,
                stepMode: "vertex",
                attributes: [
                  { shaderLocation: 0, offset: 0, format: "float32x2" },
                  { shaderLocation: 1, offset: 8, format: "float32x2" }
                ]
              }
            ]
          },
          fragment: {
            module: { __id: 2 },
            entryPoint: "fs_main",
            targets: [{ format: "rgba8unorm" }]
          }
        }
      },
      "4": { id: 4, type: "Buffer", size: 48, initialData: vbFloats.buffer },
      "5": {
        id: 5,
        type: "Texture",
        descriptor: { size: [2, 2, 1], format: "rgba8unorm" },
        data: texPixels
      },
      "6": { id: 6, type: "TextureView", texture: { __id: 5 } },
      "7": {
        id: 7,
        type: "Sampler",
        descriptor: { minFilter: "nearest", magFilter: "nearest" }
      },
      "8": {
        id: 8,
        type: "BindGroup",
        descriptor: {
          entries: [
            { binding: 0, resource: { __id: 6 } },
            { binding: 1, resource: { __id: 7 } }
          ]
        }
      },
      "9": {
        id: 9,
        type: "Texture",
        descriptor: { size: [800, 600, 1], format: "rgba8unorm" }
      },
      "10": { id: 10, type: "TextureView", texture: { __id: 9 } }
    },
    commands: [
      {
        method: "beginRenderPass",
        object: "GPUCommandEncoder#1",
        args: [
          {
            label: "main_render_pass",
            colorAttachments: [{ view: { __id: 10 } }]
          }
        ],
        result: "GPURenderPassEncoder#1"
      },
      { method: "setPipeline", object: "GPURenderPassEncoder#1", args: [{ __id: 3 }] },
      { method: "setBindGroup", object: "GPURenderPassEncoder#1", args: [0, { __id: 8 }] },
      { method: "setVertexBuffer", object: "GPURenderPassEncoder#1", args: [0, { __id: 4 }] },
      { method: "draw", object: "GPURenderPassEncoder#1", args: [3, 1, 0, 0] },
      { method: "end", object: "GPURenderPassEncoder#1", args: [] }
    ]
  };

  // Pixel (100, 75) is in the top-left region -> uv ~(0.125, 0.125) -> nearest sample is (0,0) = Red
  const session = prepareShaderDebugSession({
    capture,
    commandIndex: 4,
    stage: "fragment",
    invocation: { pixelX: 100, pixelY: 75 }
  });

  assert(session instanceof ShaderDebugSession);
  assert.equal(session.stage, "fragment");
  assert.equal(session.entryPoint, "fs_main");

  session.continueExecution();
  assert.equal(session.isAtEnd, true);

  const colRet = session.debugger.getReturnValue();
  assert(colRet, "Fragment shader should return sampled color");
  assert.deepEqual(Array.from(colRet), [1, 0, 0, 1]); // Red

  // Test fragment quad inputs generation (aligned to 2x2 even pixel boundaries)
  const quadInputs = buildFragmentQuadInputs({
    capture,
    pipeline: capture.objects["3"],
    drawCmd: capture.commands[4],
    vertexBuffers: [{ slot: 0, buffer: vbFloats.buffer }],
    pixelX: 100,
    pixelY: 75
  });

  assert.equal(quadInputs.length, 4);
  assert.equal(quadInputs[0].position[0], 100.5);
  assert.equal(quadInputs[0].position[1], 74.5);
  assert.equal(quadInputs[1].position[0], 101.5);
  assert.equal(quadInputs[1].position[1], 74.5);
  assert.equal(quadInputs[2].position[0], 100.5);
  assert.equal(quadInputs[2].position[1], 75.5);
  assert.equal(quadInputs[3].position[0], 101.5);
  assert.equal(quadInputs[3].position[1], 75.5);
});

// ---------------------------------------------------------------------------
// 5. Group 1 sparse uniforms and pageTable storage buffer resolution from bufferData
// ---------------------------------------------------------------------------
test("Group 1 sparse uniforms and pageTable storage buffer resolution from bufferData", async () => {
  const sparseUniformBytes = new Float32Array([
    -0.064, -0.064, -0.064, 0.00025, // boundsMinVoxel
    -0.052, -0.020, -0.034, 0.0,     // volumeMin
    0.052,  0.020,  0.034,  0.0,     // volumeMax
    8, 8, 8, 4                       // coarseBrick (as uint32 in same buffer)
  ]);

  const pageTableData = new Uint32Array([0, 1, 2, 3, 10, 20, 30, 40]);

  const capture = {
    objects: {
      "1": {
        id: 1,
        type: "ShaderModule",
        descriptor: {
          code: `
            struct SparseUniforms {
              boundsMinVoxel: vec4f,
              volumeMin: vec4f,
              volumeMax: vec4f,
              coarseBrick: vec4u,
            };
            @group(1) @binding(0) var<uniform> sparse: SparseUniforms;
            @group(1) @binding(1) var<storage, read> pageTable: array<u32>;
            @compute @workgroup_size(1)
            fn main() {
              let minX = sparse.volumeMin.x;
              let pt0 = pageTable[0];
            }
          `
        }
      },
      "2": {
        id: 2,
        type: "ComputePipeline",
        descriptor: {
          compute: { module: { __id: 1 }, entryPoint: "main" }
        }
      },
      "60": { id: 60, type: "Buffer", size: pageTableData.byteLength },
      "63": { id: 63, type: "Buffer", size: sparseUniformBytes.byteLength },
      "66": {
        id: 66,
        type: "BindGroup",
        descriptor: {
          entries: [
            { binding: 0, resource: { buffer: { __id: 63 } } },
            { binding: 1, resource: { buffer: { __id: 60 } } }
          ]
        }
      }
    },
    commands: [
      { method: "beginComputePass", object: "enc", result: "enc" },
      { method: "setPipeline", object: "enc", args: [{ __id: 2 }] },
      {
        method: "setBindGroup",
        object: "enc",
        args: [1, { __id: 66 }],
        bufferData: [
          { entryIndex: 0, __payloadId: 9, byteLength: sparseUniformBytes.byteLength },
          { entryIndex: 1, __payloadId: 10, byteLength: pageTableData.byteLength }
        ]
      },
      { method: "dispatchWorkgroups", object: "enc", args: [1, 1, 1] }
    ]
  };

  const payloadResolver = (id) => {
    if (id === 9) return sparseUniformBytes.buffer;
    if (id === 10) return pageTableData.buffer;
    return null;
  };

  const passState = resolvePassState(capture, 3);
  assert(passState.bindGroups[0].bufferData, "PassState must capture bufferData on setBindGroup");

  const sessionBindGroups = buildSessionBindGroups(capture, passState.bindGroups, payloadResolver);
  assert(sessionBindGroups[1][0].buffer, "Group 1 binding 0 must have buffer");
  assert(sessionBindGroups[1][0].uniform, "Group 1 binding 0 must have uniform");
  assert(sessionBindGroups[1][1].storage, "Group 1 binding 1 must have storage");

  const session = prepareShaderDebugSession({
    capture,
    commandIndex: 3,
    stage: "compute",
    entryPoint: "main",
    payloadResolver
  });

  const globals = session.getVariables({ scope: "globals" }).globals;
  assert(globals.sparse, "sparse uniforms must be populated");
  assert.equal(Number(globals.sparse.volumeMin[0].toFixed(3)), -0.052);
  assert.equal(Number(globals.sparse.volumeMax[0].toFixed(3)), 0.052);
  assert.deepEqual(globals.pageTable.slice(0, 4), [0, 1, 2, 3]);
});

// ---------------------------------------------------------------------------
// 6. Missing resource error reporting
// ---------------------------------------------------------------------------
test("Missing resource error reporting for invalid capture, missing pipeline, and missing code", async () => {
  // 1. Invalid capture
  assert.throws(
    () => prepareShaderDebugSession({ capture: null }),
    (err) => err.message.includes("Invalid capture")
  );

  // 2. Command index out of bounds
  assert.throws(
    () => prepareShaderDebugSession({ capture: { commands: [] }, commandIndex: 5 }),
    (err) => err.message.includes("out of bounds")
  );

  // 3. Missing pipeline object
  assert.throws(
    () =>
      prepareShaderDebugSession({
        capture: {
          objects: {},
          commands: [
            { method: "beginComputePass", result: "enc" },
            { method: "setPipeline", object: "enc", args: [{ __id: 999 }] },
            { method: "dispatchWorkgroups", object: "enc", args: [1, 1, 1] }
          ]
        },
        commandIndex: 2
      }),
    (err) => err.message.includes("Pipeline #999 not found")
  );

  // 4. Missing shader module object
  assert.throws(
    () =>
      prepareShaderDebugSession({
        capture: {
          objects: {
            "1": {
              id: 1,
              type: "ComputePipeline",
              descriptor: { compute: { module: { __id: 888 } } }
            }
          },
          commands: [
            { method: "beginComputePass", result: "enc" },
            { method: "setPipeline", object: "enc", args: [{ __id: 1 }] },
            { method: "dispatchWorkgroups", object: "enc", args: [1, 1, 1] }
          ]
        },
        commandIndex: 2
      }),
    (err) => err.message.includes("Shader module #888 not found")
  );

  // 5. Shader module has no WGSL code
  assert.throws(
    () =>
      prepareShaderDebugSession({
        capture: {
          objects: {
            "1": {
              id: 1,
              type: "ComputePipeline",
              descriptor: { compute: { module: { __id: 2 } } }
            },
            "2": {
              id: 2,
              type: "ShaderModule",
              descriptor: { code: "" }
            }
          },
          commands: [
            { method: "beginComputePass", result: "enc" },
            { method: "setPipeline", object: "enc", args: [{ __id: 1 }] },
            { method: "dispatchWorkgroups", object: "enc", args: [1, 1, 1] }
          ]
        },
        commandIndex: 2
      }),
    (err) => err.message.includes("contains no WGSL code")
  );
});

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------
async function runAll() {
  let passed = 0;
  let failed = 0;
  const startTime = Date.now();

  console.log(`\nRunning ${tests.length} Stage Adapters Unit Tests...\n`);

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
