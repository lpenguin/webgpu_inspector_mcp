import {
  WgslDebug,
  WgslReflect
} from "wgsl_reflect/wgsl_reflect.module.js";
import {
  ShaderDebugSession,
  decodeFloat16
} from "./shader-debug-session.js";

/**
 * Standard WebGPU vertex attribute formats configuration.
 */
export const VERTEX_FORMATS = {
  // 8-bit formats
  uint8: { type: "uint", count: 1, byteSize: 1 },
  uint8x2: { type: "uint", count: 2, byteSize: 1 },
  uint8x4: { type: "uint", count: 4, byteSize: 1 },
  sint8: { type: "sint", count: 1, byteSize: 1 },
  sint8x2: { type: "sint", count: 2, byteSize: 1 },
  sint8x4: { type: "sint", count: 4, byteSize: 1 },
  unorm8: { type: "unorm", count: 1, byteSize: 1 },
  unorm8x2: { type: "unorm", count: 2, byteSize: 1 },
  unorm8x4: { type: "unorm", count: 4, byteSize: 1 },
  snorm8: { type: "snorm", count: 1, byteSize: 1 },
  snorm8x2: { type: "snorm", count: 2, byteSize: 1 },
  snorm8x4: { type: "snorm", count: 4, byteSize: 1 },

  // 16-bit formats
  uint16: { type: "uint", count: 1, byteSize: 2 },
  uint16x2: { type: "uint", count: 2, byteSize: 2 },
  uint16x4: { type: "uint", count: 4, byteSize: 2 },
  sint16: { type: "sint", count: 1, byteSize: 2 },
  sint16x2: { type: "sint", count: 2, byteSize: 2 },
  sint16x4: { type: "sint", count: 4, byteSize: 2 },
  unorm16: { type: "unorm", count: 1, byteSize: 2 },
  unorm16x2: { type: "unorm", count: 2, byteSize: 2 },
  unorm16x4: { type: "unorm", count: 4, byteSize: 2 },
  snorm16: { type: "snorm", count: 1, byteSize: 2 },
  snorm16x2: { type: "snorm", count: 2, byteSize: 2 },
  snorm16x4: { type: "snorm", count: 4, byteSize: 2 },
  float16: { type: "float16", count: 1, byteSize: 2 },
  float16x2: { type: "float16", count: 2, byteSize: 2 },
  float16x4: { type: "float16", count: 4, byteSize: 2 },

  // 32-bit formats
  float32: { type: "float32", count: 1, byteSize: 4 },
  float32x1: { type: "float32", count: 1, byteSize: 4 },
  float32x2: { type: "float32", count: 2, byteSize: 4 },
  float32x3: { type: "float32", count: 3, byteSize: 4 },
  float32x4: { type: "float32", count: 4, byteSize: 4 },
  uint32: { type: "uint", count: 1, byteSize: 4 },
  uint32x1: { type: "uint", count: 1, byteSize: 4 },
  uint32x2: { type: "uint", count: 2, byteSize: 4 },
  uint32x3: { type: "uint", count: 3, byteSize: 4 },
  uint32x4: { type: "uint", count: 4, byteSize: 4 },
  sint32: { type: "sint", count: 1, byteSize: 4 },
  sint32x1: { type: "sint", count: 1, byteSize: 4 },
  sint32x2: { type: "sint", count: 2, byteSize: 4 },
  sint32x3: { type: "sint", count: 3, byteSize: 4 },
  sint32x4: { type: "sint", count: 4, byteSize: 4 },

  // Packed format
  "unorm10-10-10-2": { type: "unorm10_10_10_2", count: 4, byteSize: 4 },
  "unorm10_10_10_2": { type: "unorm10_10_10_2", count: 4, byteSize: 4 }
};

/**
 * Extracts numeric ID from various object reference shapes (__id, id, "#1", 1).
 */
export function refId(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const num = Number(value);
    if (!isNaN(num)) return num;
    const match = /#(\d+)/.exec(value);
    if (match) return Number(match[1]);
    return null;
  }
  if (typeof value === "object") {
    if (typeof value.__id === "number") return value.__id;
    if (typeof value.id === "number") return value.id;
    if (typeof value.id === "string" && !isNaN(Number(value.id))) return Number(value.id);
  }
  return null;
}

/**
 * Normalizes input data into an ArrayBuffer.
 */
export function toArrayBuffer(data) {
  if (!data) return new ArrayBuffer(0);
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  if (typeof data === "string") {
    const buf = Buffer.from(data, "base64");
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  if (Array.isArray(data)) {
    const isFloat = data.some((v) => typeof v === "number" && (!Number.isInteger(v) || v < 0 || v > 255));
    if (isFloat) {
      return new Float32Array(data).buffer;
    }
    return new Uint8Array(data).buffer;
  }
  if (data.bytes) {
    return toArrayBuffer(data.bytes);
  }
  if (data.base64 || data.__base64) {
    return toArrayBuffer(data.base64 || data.__base64);
  }
  return new ArrayBuffer(0);
}

/**
 * Resolves a payload reference or payload ID into raw bytes (Buffer/Uint8Array/ArrayBuffer).
 */
export function resolvePayloadBytes(ref, resolver, capture = null) {
  if (!ref) return null;
  if (ref instanceof ArrayBuffer || ArrayBuffer.isView(ref) || Buffer.isBuffer(ref)) {
    return ref;
  }

  const payloadId =
    typeof ref === "number"
      ? ref
      : (ref.__payloadId ?? ref.payloadId ?? (typeof ref.id === "number" ? ref.id : null));

  if (payloadId !== null && payloadId !== undefined) {
    if (typeof resolver === "function") {
      const res = resolver(payloadId);
      if (res) return res;
    } else if (resolver instanceof Map && resolver.has(payloadId)) {
      return resolver.get(payloadId);
    } else if (resolver && typeof resolver === "object" && resolver[payloadId]) {
      return resolver[payloadId];
    }

    const payloads = capture?.payloads || capture?.metadata?.payloads;
    if (payloads) {
      if (payloads instanceof Map && payloads.has(payloadId)) {
        return payloads.get(payloadId);
      }
      if (typeof payloads === "object" && payloads[payloadId]) {
        return payloads[payloadId];
      }
    }
  }

  if (ref.bytes) return ref.bytes;
  const b64 = ref.base64 || ref.__base64;
  if (typeof b64 === "string") {
    return Buffer.from(b64, "base64");
  }
  if (Array.isArray(ref.data) || Array.isArray(ref)) {
    const arr = ref.data || ref;
    const isFloat = arr.some((v) => typeof v === "number" && (!Number.isInteger(v) || v < 0 || v > 255));
    if (isFloat) {
      return new Float32Array(arr);
    }
    return new Uint8Array(arr);
  }
  return null;
}

/**
 * Assembles all recorded buffer bytes from commands or payloads.
 */
export function resolveBufferData(bufId, bufRec, entry, capture, payloadResolver) {
  if (entry?.data) return entry.data;
  if (entry?.bytes) return entry.bytes;
  if (entry?.base64 || entry?.__base64) {
    return Buffer.from(entry.base64 || entry.__base64, "base64");
  }

  if (bufRec?.data) return bufRec.data;
  if (bufRec?.bytes) return bufRec.bytes;
  if (bufRec?.initialData) return bufRec.initialData;
  if (bufRec?.base64 || bufRec?.__base64) {
    return Buffer.from(bufRec.base64 || bufRec.__base64, "base64");
  }

  const payloadId =
    entry?.__payloadId ?? entry?.payloadId ?? bufRec?.__payloadId ?? bufRec?.payloadId;
  if (payloadId !== undefined && payloadId !== null) {
    const res = resolvePayloadBytes({ __payloadId: payloadId }, payloadResolver, capture);
    if (res) return res;
  }

  // Scan writeBuffer commands in capture if present
  const commands = capture?.commands || capture?.metadata?.commands || [];
  let assembled = null;
  const size = bufRec?.size || 0;

  for (const cmd of commands) {
    if (!cmd) continue;
    if (cmd.method === "writeBuffer" && refId(cmd.args?.[0]) === bufId) {
      if (!assembled) {
        assembled = new Uint8Array(size || 65536);
      }
      const dstOffset = Number(cmd.args[1]) || 0;
      let writtenBytes = null;
      if (cmd.bufferData && Array.isArray(cmd.bufferData) && cmd.bufferData.length > 0) {
        const e = cmd.bufferData[0];
        writtenBytes = resolvePayloadBytes(e, payloadResolver, capture);
      } else if (cmd.args[2]) {
        writtenBytes = resolvePayloadBytes(cmd.args[2], payloadResolver, capture) || cmd.args[2];
      }
      if (writtenBytes) {
        const arr = new Uint8Array(toArrayBuffer(writtenBytes));
        if (dstOffset + arr.length > assembled.length) {
          const bigger = new Uint8Array(dstOffset + arr.length);
          bigger.set(assembled);
          assembled = bigger;
        }
        assembled.set(arr, dstOffset);
      }
    }
  }

  if (assembled) return assembled.buffer;

  if (typeof payloadResolver === "function" && bufId != null) {
    const res = payloadResolver(bufId);
    if (res) return res;
  }

  return new ArrayBuffer(Math.max(size || 0, 256));
}

/**
 * Resolves texture pixel data payloads from capture and resolver.
 */
export function resolveTextureData(texObj, viewObj, entry, capture, payloadResolver) {
  if (entry?.texture) return entry.texture;
  if (texObj?.data) return texObj.data;
  if (texObj?.bytes) return texObj.bytes;
  if (texObj?.mipData && Array.isArray(texObj.mipData)) {
    const mips = [];
    for (const mip of texObj.mipData) {
      const bytes = resolvePayloadBytes(mip, payloadResolver, capture);
      if (bytes) {
        mips.push(new Uint8Array(toArrayBuffer(bytes)));
      }
    }
    if (mips.length === 1) return mips[0];
    if (mips.length > 1) return mips;
  }
  const payloadId = texObj?.__payloadId ?? texObj?.payloadId;
  if (payloadId != null) {
    const bytes = resolvePayloadBytes({ __payloadId: payloadId }, payloadResolver, capture);
    if (bytes) return new Uint8Array(toArrayBuffer(bytes));
  }
  return new Uint8Array(0);
}

/**
 * Reconstructs bind groups for WgslDebug from captured setBindGroup commands and objects.
 *
 * @param {Object} capture
 * @param {Array|Object} bindGroupCommands
 * @param {Function|Map|Object} [payloadResolver]
 * @returns {Record<string, Record<string, any>>}
 */
export function buildSessionBindGroups(capture, bindGroupCommands = [], payloadResolver = null) {
  const objects = capture?.metadata?.objects || capture?.objects || {};
  const sessionBindGroups = {};

  let list = [];
  if (Array.isArray(bindGroupCommands)) {
    list = bindGroupCommands;
  } else if (bindGroupCommands && typeof bindGroupCommands === "object") {
    list = Object.values(bindGroupCommands);
  }

  for (const bgCmd of list) {
    if (!bgCmd) continue;
    let groupIndex = null;
    let bindGroupId = null;
    let dynamicOffsets = bgCmd.dynamicOffsets || [];

    if (bgCmd.group !== undefined) {
      groupIndex = Number(bgCmd.group);
    } else if (bgCmd.slot !== undefined) {
      groupIndex = Number(bgCmd.slot);
    } else if (Array.isArray(bgCmd.args)) {
      groupIndex = Number(bgCmd.args[0]);
    }

    if (bgCmd.bindGroupId !== undefined) {
      bindGroupId = refId(bgCmd.bindGroupId);
    } else if (bgCmd.id !== undefined) {
      bindGroupId = refId(bgCmd.id);
    } else if (Array.isArray(bgCmd.args) && bgCmd.args[1] !== undefined) {
      bindGroupId = refId(bgCmd.args[1]);
    }

    if (groupIndex === null) continue;
    if (sessionBindGroups[groupIndex] === undefined) {
      sessionBindGroups[groupIndex] = {};
    }

    if (bindGroupId === null) continue;
    const bgRec = objects[String(bindGroupId)] || objects[bindGroupId];
    if (!bgRec) continue;

    const entries = bgRec.descriptor?.entries || bgRec.entries || [];
    let dynamicOffsetIdx = 0;

    for (const entry of entries) {
      const binding = entry.binding !== undefined ? Number(entry.binding) : 0;

      // 1. Buffer Resource
      const isBuffer =
        entry.resource?.buffer !== undefined ||
        entry.buffer !== undefined ||
        (entry.resource && objects[String(refId(entry.resource))]?.type === "Buffer");

      if (isBuffer) {
        const bufRef = entry.resource?.buffer || entry.buffer || entry.resource;
        const bufId = refId(bufRef);
        const bufRec = bufId != null ? (objects[String(bufId)] || objects[bufId]) : null;

        let offset = entry.resource?.offset ?? entry.offset ?? 0;
        let size = entry.resource?.size ?? entry.size ?? null;

        if (dynamicOffsets && dynamicOffsets.length > dynamicOffsetIdx) {
          offset += Number(dynamicOffsets[dynamicOffsetIdx++]) || 0;
        }

        const rawBuf = resolveBufferData(bufId, bufRec, entry, capture, payloadResolver);
        let arrayBuf = toArrayBuffer(rawBuf);

        if (bufRec?.size && arrayBuf.byteLength < bufRec.size) {
          const expanded = new ArrayBuffer(bufRec.size);
          new Uint8Array(expanded).set(new Uint8Array(arrayBuf));
          arrayBuf = expanded;
        }

        let slicedBuf = arrayBuf;
        if (offset > 0 || size !== null) {
          const start = Math.min(offset, arrayBuf.byteLength);
          const end = size !== null ? Math.min(start + size, arrayBuf.byteLength) : arrayBuf.byteLength;
          slicedBuf = arrayBuf.slice(start, end);
        }
        if (slicedBuf.byteLength === 0) {
          slicedBuf = new ArrayBuffer(256);
        }

        sessionBindGroups[groupIndex][binding] = {
          uniform: slicedBuf,
          storage: slicedBuf,
          buffer: slicedBuf,
          descriptor: {
            size: size ?? (bufRec?.size || arrayBuf.byteLength),
            offset
          }
        };
        continue;
      }

      // 2. Texture / TextureView Resource
      const isTextureView =
        entry.resource?.texture !== undefined ||
        entry.resource?.view !== undefined ||
        entry.texture !== undefined ||
        entry.view !== undefined ||
        (entry.resource && (
          objects[String(refId(entry.resource))]?.type === "TextureView" ||
          objects[String(refId(entry.resource))]?.type === "Texture"
        ));

      if (isTextureView) {
        const resRef =
          entry.resource?.texture ||
          entry.resource?.view ||
          entry.texture ||
          entry.view ||
          entry.resource;
        const resId = refId(resRef);
        const resObj = resId != null ? (objects[String(resId)] || objects[resId]) : null;

        let texObj = null;
        let viewDesc = null;
        let texDesc = null;

        if (resObj?.type === "TextureView") {
          const texRef = resObj.texture || resObj.textureId || resObj.__id;
          const texId = refId(texRef);
          texObj = texId != null ? (objects[String(texId)] || objects[texId]) : null;
          viewDesc = resObj.descriptor || resObj;
          texDesc = texObj?.descriptor || texObj || {};
        } else {
          texObj = resObj;
          texDesc = texObj?.descriptor || texObj || {};
          viewDesc = null;
        }

        const texData = resolveTextureData(texObj, resObj, entry, capture, payloadResolver);

        sessionBindGroups[groupIndex][binding] = {
          texture: texData,
          descriptor: texDesc,
          view: viewDesc
        };
        continue;
      }

      // 3. Sampler Resource
      const isSampler =
        entry.resource?.sampler !== undefined ||
        entry.sampler !== undefined ||
        (entry.resource && objects[String(refId(entry.resource))]?.type === "Sampler");

      if (isSampler) {
        const sampRef = entry.resource?.sampler || entry.sampler || entry.resource;
        const sampId = refId(sampRef);
        const sampRec = sampId != null ? (objects[String(sampId)] || objects[sampId]) : null;
        const sampDesc = sampRec?.descriptor || sampRec || entry.sampler || {};

        sessionBindGroups[groupIndex][binding] = {
          sampler: sampDesc
        };
        continue;
      }

      // Direct fallback
      if (entry.uniform || entry.storage) {
        const raw = entry.uniform || entry.storage;
        sessionBindGroups[groupIndex][binding] = {
          uniform: toArrayBuffer(raw),
          storage: toArrayBuffer(raw)
        };
      } else if (entry.texture && entry.descriptor) {
        sessionBindGroups[groupIndex][binding] = {
          texture: entry.texture,
          descriptor: entry.descriptor,
          view: entry.view || null
        };
      } else if (entry.sampler) {
        sessionBindGroups[groupIndex][binding] = {
          sampler: entry.sampler
        };
      }
    }
  }

  return sessionBindGroups;
}

/**
 * Reads a scalar from DataView based on component type and byte size.
 */
export function readScalar(view, offset, type, size) {
  if (offset + size > view.byteLength) return null;
  if (type === "float32") {
    return view.getFloat32(offset, true);
  }
  if (type === "float16") {
    return decodeFloat16(view.getUint16(offset, true));
  }
  let raw;
  const signed = type === "sint" || type === "snorm";
  if (size === 1) {
    raw = signed ? view.getInt8(offset) : view.getUint8(offset);
  } else if (size === 2) {
    raw = signed ? view.getInt16(offset, true) : view.getUint16(offset, true);
  } else {
    raw = signed ? view.getInt32(offset, true) : view.getUint32(offset, true);
  }

  if (type === "unorm") {
    return raw / (Math.pow(2, size * 8) - 1);
  }
  if (type === "snorm") {
    return Math.max(raw / (Math.pow(2, size * 8 - 1) - 1), -1);
  }
  return raw;
}

/**
 * Decodes a single vertex attribute according to its WebGPU format.
 */
export function decodeVertexAttribute(view, offset, format) {
  const fmt = VERTEX_FORMATS[format];
  if (!fmt) return undefined;

  if (fmt.type === "unorm10_10_10_2") {
    if (offset + 4 > view.byteLength) return undefined;
    const u32 = view.getUint32(offset, true);
    return [
      (u32 & 0x3ff) / 1023.0,
      ((u32 >>> 10) & 0x3ff) / 1023.0,
      ((u32 >>> 20) & 0x3ff) / 1023.0,
      ((u32 >>> 30) & 0x3) / 3.0
    ];
  }

  if (offset + fmt.count * fmt.byteSize > view.byteLength) {
    return undefined;
  }

  if (fmt.count === 1) {
    return readScalar(view, offset, fmt.type, fmt.byteSize);
  }

  const comps = [];
  for (let c = 0; c < fmt.count; c++) {
    comps.push(readScalar(view, offset + c * fmt.byteSize, fmt.type, fmt.byteSize));
  }
  return comps;
}

/**
 * Decodes vertex inputs for vertex shader execution.
 *
 * @param {Object} pipelineDesc - Pipeline descriptor with vertex.buffers
 * @param {Array|Object} vertexBufferData - Array or map of vertex buffers
 * @param {number} vertexIndex - Vertex index
 * @param {number} instanceIndex - Instance index
 * @param {Object} [shaderInputs] - Additional shader inputs
 * @returns {Record<string, any>}
 */
export function fetchVertexInputs(
  pipelineDesc,
  vertexBufferData = [],
  vertexIndex = 0,
  instanceIndex = 0,
  shaderInputs = {}
) {
  const vIndex = Number(vertexIndex) || 0;
  const instIndex = Number(instanceIndex) || 0;
  const inputs = {
    vertex_index: vIndex,
    instance_index: instIndex,
    vertexIndex: vIndex,
    instanceIndex: instIndex,
    ...shaderInputs
  };

  const bufferLayouts =
    pipelineDesc?.vertex?.buffers ||
    pipelineDesc?.buffers ||
    (Array.isArray(pipelineDesc) ? pipelineDesc : []);

  for (let slot = 0; slot < bufferLayouts.length; slot++) {
    const layout = bufferLayouts[slot];
    if (!layout || !Array.isArray(layout.attributes)) continue;

    let rawBuf = null;
    let boundOffset = 0;

    if (Array.isArray(vertexBufferData)) {
      const entry =
        vertexBufferData.find((vb) => vb && (vb.slot === slot || vb.slot === String(slot))) ||
        vertexBufferData[slot];
      if (entry) {
        if (entry.buffer) {
          rawBuf = entry.buffer;
          boundOffset = entry.offset || 0;
        } else if (entry.bytes) {
          rawBuf = entry.bytes;
          boundOffset = entry.offset || 0;
        } else {
          rawBuf = entry;
        }
      }
    } else if (vertexBufferData && typeof vertexBufferData === "object") {
      const entry = vertexBufferData[slot] || vertexBufferData[String(slot)];
      if (entry) {
        if (entry.buffer) {
          rawBuf = entry.buffer;
          boundOffset = entry.offset || 0;
        } else if (entry.bytes) {
          rawBuf = entry.bytes;
          boundOffset = entry.offset || 0;
        } else {
          rawBuf = entry;
        }
      }
    }

    if (!rawBuf) continue;

    const arrayBuf = toArrayBuffer(rawBuf);
    const view = new DataView(arrayBuf);
    const stepMode = layout.stepMode || "vertex";
    const stride = layout.arrayStride || 0;
    const itemIndex = stepMode === "instance" ? instIndex : vIndex;
    const baseOffset = boundOffset + itemIndex * stride;

    for (const at of layout.attributes) {
      const attrOffset = baseOffset + (at.offset || 0);
      const loc = at.shaderLocation;
      const decoded = decodeVertexAttribute(view, attrOffset, at.format);
      if (decoded !== undefined && decoded !== null) {
        inputs[loc] = decoded;
        inputs[String(loc)] = decoded;
      }
    }
  }

  return inputs;
}

/**
 * Reconstructs triangles and interpolates varyings and @builtin(position) at pixel (pixelX, pixelY).
 */
export function prepareFragmentInputs(params = {}) {
  const {
    capture = null,
    pipeline = null,
    drawCmd = null,
    vertexBuffers = [],
    indexBuffer = null,
    renderPass = null,
    pixelX = 0,
    pixelY = 0,
    invocation = {},
    payloadResolver = null
  } = params;

  const objects = capture?.metadata?.objects || capture?.objects || {};
  const px = Number(invocation.pixelX ?? invocation.x ?? pixelX) || 0;
  const py = Number(invocation.pixelY ?? invocation.y ?? pixelY) || 0;

  // Resolve target dimensions
  let targetWidth = 800;
  let targetHeight = 600;

  if (renderPass?.colorAttachments?.[0]?.view) {
    const viewId = refId(renderPass.colorAttachments[0].view);
    const viewRec = viewId != null ? objects[String(viewId)] : null;
    const texId = viewRec ? refId(viewRec.texture || viewRec.textureId || viewRec.__id) : null;
    const texRec = texId != null ? objects[String(texId)] : null;
    if (texRec?.descriptor?.size) {
      targetWidth = texRec.descriptor.size[0] || targetWidth;
      targetHeight = texRec.descriptor.size[1] || targetHeight;
    } else if (texRec?.width && texRec?.height) {
      targetWidth = texRec.width;
      targetHeight = texRec.height;
    }
  }

  if (invocation.targetWidth) targetWidth = invocation.targetWidth;
  if (invocation.targetHeight) targetHeight = invocation.targetHeight;

  // Default fragment inputs
  const defaultInputs = {
    position: [px + 0.5, py + 0.5, 0.5, 1.0],
    front_facing: invocation.frontFacing ?? true,
    sample_index: invocation.sampleIndex ?? 0,
    sample_mask: invocation.sampleMask ?? 0xffffffff,
    ...(invocation.inputs || {}),
    ...invocation
  };

  // If no pipeline or vertex stage exists, return defaults
  const pipeDesc = pipeline?.descriptor || pipeline;
  if (!pipeDesc || !pipeDesc.vertex) {
    return defaultInputs;
  }

  const vModuleId = refId(pipeDesc.vertex.module);
  const vShaderObj = vModuleId != null ? objects[String(vModuleId)] : null;
  const vCode = vShaderObj?.descriptor?.code || vShaderObj?.code;
  if (!vCode) {
    return defaultInputs;
  }

  const vEntryPoint = pipeDesc.vertex.entryPoint || "main";
  const instanceIndex = Number(invocation.instanceIndex) || 0;

  // Reconstruct index array
  let indices = [];
  const method = drawCmd?.method || "draw";
  const drawArgs = drawCmd?.args || [];

  if (method === "drawIndexed" || indexBuffer) {
    const indexCount = Number(drawArgs[0]) || 0;
    const firstIndex = Number(drawArgs[2]) || 0;
    const baseVertex = Number(drawArgs[3]) || 0;

    let idxBuf = indexBuffer?.buffer ? toArrayBuffer(indexBuffer.buffer) : null;
    if (!idxBuf && indexBuffer?.bufferId != null) {
      const bData = resolveBufferData(
        indexBuffer.bufferId,
        objects[String(indexBuffer.bufferId)],
        null,
        capture,
        payloadResolver
      );
      idxBuf = toArrayBuffer(bData);
    }

    if (idxBuf && idxBuf.byteLength > 0) {
      const view = new DataView(idxBuf);
      const is32 = indexBuffer?.format === "uint32";
      const stride = is32 ? 4 : 2;
      const offset = Number(indexBuffer?.offset) || 0;
      for (let i = 0; i < indexCount; i++) {
        const off = offset + (firstIndex + i) * stride;
        if (off + stride <= idxBuf.byteLength) {
          const idx = is32 ? view.getUint32(off, true) : view.getUint16(off, true);
          indices.push(idx + baseVertex);
        }
      }
    } else {
      for (let i = 0; i < indexCount; i++) {
        indices.push(firstIndex + i + baseVertex);
      }
    }
  } else {
    const vertexCount = Number(drawArgs[0]) || 3;
    const firstVertex = Number(drawArgs[2]) || 0;
    for (let i = 0; i < vertexCount; i++) {
      indices.push(firstVertex + i);
    }
  }

  // Build vertex bind groups
  const vBindGroups = buildSessionBindGroups(capture, params.bindGroups || [], payloadResolver);

  // Execute vertex shader for all vertices
  const vertexCache = new Map();
  const vReflect = new WgslReflect(vCode);
  const vFnInfo = vReflect.getFunctionInfo(vEntryPoint);

  for (const vIdx of indices) {
    if (vertexCache.has(vIdx)) continue;
    const vInputs = fetchVertexInputs(pipeDesc, vertexBuffers, vIdx, instanceIndex);
    const vDebug = new WgslDebug(vCode);
    const ok = vDebug.debugVertex(vEntryPoint, vInputs, vBindGroups);
    if (!ok) continue;

    while (vDebug.stepNext(true)) {}
    const ret = vDebug.getReturnValue();
    if (!ret) continue;

    let pos = ret.position || ret.pos || ret["@position"] || [0, 0, 0, 1];
    if (!Array.isArray(pos) && pos?.data) {
      pos = Array.from(pos.data);
    }
    if (Array.isArray(pos) && pos.length === 2) {
      pos = [pos[0], pos[1], 0.0, 1.0];
    } else if (Array.isArray(pos) && pos.length === 3) {
      pos = [pos[0], pos[1], pos[2], 1.0];
    }

    // Map vertex output member locations if struct return
    const varyings = { ...ret };
    if (vFnInfo?.returnType?.members) {
      for (const m of vFnInfo.returnType.members) {
        const locAttr = m.attributes?.find((a) => a.name === "location");
        if (locAttr && locAttr.value !== undefined && ret[m.name] !== undefined) {
          const loc = Number(locAttr.value);
          varyings[loc] = ret[m.name];
          varyings[String(loc)] = ret[m.name];
        }
      }
    }

    vertexCache.set(vIdx, { position: pos, varyings });
  }

  // Build primitives triangles
  const topology = pipeDesc.primitive?.topology || "triangle-list";
  const frontFace = pipeDesc.primitive?.frontFace || "ccw";
  const cullMode = pipeDesc.primitive?.cullMode || "none";

  const triangles = [];
  if (topology === "triangle-strip") {
    for (let i = 0; i + 2 < indices.length; i++) {
      if (i % 2 === 0) {
        triangles.push([indices[i], indices[i + 1], indices[i + 2]]);
      } else {
        triangles.push([indices[i + 1], indices[i], indices[i + 2]]);
      }
    }
  } else {
    // triangle-list
    for (let i = 0; i + 2 < indices.length; i += 3) {
      triangles.push([indices[i], indices[i + 1], indices[i + 2]]);
    }
  }

  const Px = px + 0.5;
  const Py = py + 0.5;

  for (const [i0, i1, i2] of triangles) {
    const v0 = vertexCache.get(i0);
    const v1 = vertexCache.get(i1);
    const v2 = vertexCache.get(i2);
    if (!v0 || !v1 || !v2) continue;

    const p0 = v0.position;
    const p1 = v1.position;
    const p2 = v2.position;

    const w0 = p0[3] || 1.0;
    const w1 = p1[3] || 1.0;
    const w2 = p2[3] || 1.0;

    const x0_ndc = p0[0] / w0;
    const y0_ndc = p0[1] / w0;
    const z0_ndc = p0[2] / w0;

    const x1_ndc = p1[0] / w1;
    const y1_ndc = p1[1] / w1;
    const z1_ndc = p1[2] / w1;

    const x2_ndc = p2[0] / w2;
    const y2_ndc = p2[1] / w2;
    const z2_ndc = p2[2] / w2;

    const sx0 = (x0_ndc + 1) * 0.5 * targetWidth;
    const sy0 = (1 - y0_ndc) * 0.5 * targetHeight;

    const sx1 = (x1_ndc + 1) * 0.5 * targetWidth;
    const sy1 = (1 - y1_ndc) * 0.5 * targetHeight;

    const sx2 = (x2_ndc + 1) * 0.5 * targetWidth;
    const sy2 = (1 - y2_ndc) * 0.5 * targetHeight;

    const det = (sy1 - sy2) * (sx0 - sx2) + (sx2 - sx1) * (sy0 - sy2);
    if (Math.abs(det) < 1e-8) continue;

    const l0 = ((sy1 - sy2) * (Px - sx2) + (sx2 - sx1) * (Py - sy2)) / det;
    const l1 = ((sy2 - sy0) * (Px - sx2) + (sx0 - sx2) * (Py - sy2)) / det;
    const l2 = 1.0 - l0 - l1;

    const eps = -1e-4;
    if (l0 >= eps && l1 >= eps && l2 >= eps) {
      const area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
      const isCcw = frontFace === "ccw";
      const isFront = isCcw ? area < 0 : area > 0;

      if (cullMode === "back" && !isFront) continue;
      if (cullMode === "front" && isFront) continue;

      const invW = l0 / w0 + l1 / w1 + l2 / w2;
      const wP = 1.0 / invW;
      const zP = (l0 * z0_ndc / w0 + l1 * z1_ndc / w1 + l2 * z2_ndc / w2) * wP;

      const resultInputs = {
        position: [Px, Py, zP, invW],
        front_facing: isFront,
        sample_index: 0,
        sample_mask: 0xffffffff
      };

      for (const key of Object.keys(v0.varyings)) {
        if (key === "pos" || key === "position" || key === "@position") continue;
        const val0 = v0.varyings[key];
        const val1 = v1.varyings[key];
        const val2 = v2.varyings[key];

        if (typeof val0 === "number") {
          resultInputs[key] = (l0 * val0 / w0 + l1 * (val1 ?? 0) / w1 + l2 * (val2 ?? 0) / w2) * wP;
        } else if (Array.isArray(val0)) {
          resultInputs[key] = val0.map(
            (c, idx) => (l0 * c / w0 + l1 * (val1?.[idx] ?? 0) / w1 + l2 * (val2?.[idx] ?? 0) / w2) * wP
          );
        }
      }

      return Object.assign(resultInputs, invocation.inputs || {});
    }
  }

  return defaultInputs;
}

/**
 * Builds 4 stage inputs corresponding to a 2x2 fragment quad around (pixelX, pixelY).
 *
 * @param {Object} params
 * @returns {[Object, Object, Object, Object]} 4 quad lane inputs
 */
export function buildFragmentQuadInputs(params = {}) {
  const baseX = Number(params.invocation?.pixelX ?? params.invocation?.x ?? params.pixelX) || 0;
  const baseY = Number(params.invocation?.pixelY ?? params.invocation?.y ?? params.pixelY) || 0;

  const offsets = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
  ];

  return offsets.map(([dx, dy]) => {
    return prepareFragmentInputs({
      ...params,
      pixelX: baseX + dx,
      pixelY: baseY + dy,
      invocation: {
        ...(params.invocation || {}),
        pixelX: baseX + dx,
        pixelY: baseY + dy,
        x: baseX + dx,
        y: baseY + dy
      }
    });
  });
}

/**
 * Resolves pass state by walking backwards from a draw / dispatch command.
 */
export function resolvePassState(capture, targetIndex) {
  const commands = capture?.metadata?.commands || capture?.commands || [];
  const targetCmd = commands[targetIndex];
  if (!targetCmd) return null;

  const encoderId = targetCmd.object;
  let pipeline = null;
  const bindGroups = {};
  const vertexBuffers = {};
  let indexBuffer = null;
  let renderPass = null;

  for (let i = targetIndex - 1; i >= 0; i--) {
    const c = commands[i];
    if (!c) continue;

    if (
      (c.method === "beginRenderPass" || c.method === "beginComputePass") &&
      (c.result === encoderId || c.object === encoderId || !encoderId)
    ) {
      renderPass = {
        commandIndex: i,
        method: c.method,
        colorAttachments: c.args?.[0]?.colorAttachments || [],
        depthStencilAttachment: c.args?.[0]?.depthStencilAttachment || null,
        label: c.args?.[0]?.label || ""
      };
      break;
    }

    if (encoderId && c.object !== encoderId) continue;

    if (c.method === "setPipeline" && pipeline === null) {
      pipeline = {
        commandIndex: i,
        id: refId(c.args?.[0])
      };
    } else if (c.method === "setBindGroup") {
      const slot = c.args?.[0];
      if (slot !== undefined && bindGroups[slot] === undefined) {
        bindGroups[slot] = {
          commandIndex: i,
          group: slot,
          bindGroupId: refId(c.args?.[1]),
          dynamicOffsets: c.args?.[2] || []
        };
      }
    } else if (c.method === "setVertexBuffer") {
      const slot = c.args?.[0];
      if (slot !== undefined && vertexBuffers[slot] === undefined) {
        bindBuffersSlot(vertexBuffers, slot, c, i);
      }
    } else if (c.method === "setIndexBuffer" && indexBuffer === null) {
      indexBuffer = {
        commandIndex: i,
        bufferId: refId(c.args?.[0]),
        format: c.args?.[1] || "uint16",
        offset: c.args?.[2] || 0,
        size: c.args?.[3] || null,
        bufferData: c.bufferData
      };
    }
  }

  return {
    commandIndex: targetIndex,
    command: targetCmd,
    encoderId,
    pipeline,
    bindGroups: Object.values(bindGroups).sort((a, b) => a.group - b.group),
    vertexBuffers: Object.values(vertexBuffers).sort((a, b) => a.slot - b.slot),
    indexBuffer,
    renderPass
  };
}

function bindBuffersSlot(vertexBuffers, slot, c, commandIndex) {
  vertexBuffers[slot] = {
    slot,
    commandIndex,
    bufferId: refId(c.args?.[1]),
    offset: c.args?.[2] || 0,
    size: c.args?.[3] || null,
    bufferData: c.bufferData
  };
}

/**
 * Prepares and initializes a ShaderDebugSession from a WebGPU capture and command index.
 *
 * @param {Object} params
 * @param {Object} params.capture - WebGPU capture
 * @param {number} [params.commandIndex] - Index of draw / dispatch command
 * @param {"compute"|"vertex"|"fragment"} [params.stage] - Shader stage
 * @param {string} [params.entryPoint] - Entry point name
 * @param {Object} [params.invocation] - Invocation coordinates / vertex index / pixel coords
 * @param {Function|Map|Object} [params.payloadResolver] - Payload resolver
 * @param {Object} [params.options] - Session options / pipeline constants
 * @param {string} [params.sessionId] - Custom session ID
 * @returns {ShaderDebugSession} Initialized ShaderDebugSession
 */
export function prepareShaderDebugSession(params = {}) {
  const {
    capture,
    stage: requestedStage,
    entryPoint: requestedEntryPoint,
    invocation = {},
    payloadResolver = null,
    options = {},
    sessionId
  } = params;

  if (!capture || typeof capture !== "object") {
    throw new Error("Invalid capture: capture object is required");
  }

  const objects = capture?.metadata?.objects || capture?.objects || {};
  const commands = capture?.metadata?.commands || capture?.commands || [];

  // 1. Locate command index if not provided
  let commandIndex = params.commandIndex;
  if (commandIndex === undefined || commandIndex === null) {
    if (requestedStage === "compute") {
      commandIndex = commands.findIndex(
        (c) => c && (c.method === "dispatchWorkgroups" || c.method === "dispatchWorkgroupsIndirect")
      );
    } else if (requestedStage === "vertex" || requestedStage === "fragment") {
      commandIndex = commands.findIndex(
        (c) => c && (c.method === "draw" || c.method === "drawIndexed" || c.method === "drawIndirect")
      );
    } else {
      commandIndex = commands.findIndex(
        (c) =>
          c &&
          (c.method === "draw" ||
            c.method === "drawIndexed" ||
            c.method === "dispatchWorkgroups" ||
            c.method === "dispatchWorkgroupsIndirect")
      );
    }
  }

  if (commandIndex < 0 || commandIndex >= commands.length) {
    if (params.code) {
      const code = params.code;
      const stage = requestedStage || "compute";
      const entryPoint = requestedEntryPoint || "main";
      let bgList = params.bindGroups || [];
      if (!bgList || (Array.isArray(bgList) && bgList.length === 0)) {
        const bgEntries = [];
        for (const [id, obj] of Object.entries(objects)) {
          if (obj && (obj.type === "BindGroup" || obj.descriptor?.entries)) {
            bgEntries.push({ bindGroupId: Number(id) || id, group: bgEntries.length });
          }
        }
        bgList = bgEntries;
      }
      const sessionBindGroups = buildSessionBindGroups(capture, bgList, payloadResolver);
      const session = new ShaderDebugSession({
        id: sessionId,
        code,
        stage,
        entryPoint,
        invocation,
        bindGroups: sessionBindGroups,
        options: {
          constants: options.constants,
          ...options
        },
        stageConfig: {}
      });
      session.init();
      return session;
    }
    throw new Error(`Command index ${commandIndex} is out of bounds (commands count: ${commands.length})`);
  }

  const passState = resolvePassState(capture, commandIndex);
  if (!passState) {
    throw new Error(`Failed to resolve pass state for command #${commandIndex}`);
  }

  // 2. Resolve pipeline
  let pipelineId = passState.pipeline?.id;
  if (pipelineId == null && options.pipelineId != null) {
    pipelineId = options.pipelineId;
  }
  if (pipelineId == null) {
    // Search objects for any pipeline
    const pEntry = Object.values(objects).find(
      (o) => o && (o.type === "RenderPipeline" || o.type === "ComputePipeline")
    );
    if (pEntry) pipelineId = pEntry.id;
  }

  if (pipelineId == null) {
    throw new Error(`No pipeline bound for command #${commandIndex}`);
  }

  const pipelineObj = objects[String(pipelineId)] || objects[pipelineId];
  if (!pipelineObj) {
    throw new Error(`Pipeline #${pipelineId} not found in capture objects`);
  }

  const pipeDesc = pipelineObj.descriptor || pipelineObj;

  // 3. Determine stage
  let stage = requestedStage;
  if (!stage) {
    if (pipelineObj.type === "ComputePipeline" || pipeDesc.compute) {
      stage = "compute";
    } else if (pipelineObj.type === "RenderPipeline" || pipeDesc.vertex) {
      stage = "vertex";
    } else {
      stage = "compute";
    }
  }

  // 4. Resolve shader module and entry point
  let stageDesc = null;
  if (stage === "compute") {
    stageDesc = pipeDesc.compute;
  } else if (stage === "vertex") {
    stageDesc = pipeDesc.vertex;
  } else if (stage === "fragment") {
    stageDesc = pipeDesc.fragment;
  }

  if (!stageDesc && !params.code && !options?.code) {
    throw new Error(`Pipeline #${pipelineId} has no descriptor for stage "${stage}"`);
  }

  const moduleId = stageDesc ? refId(stageDesc.module) : null;
  let code = params.code || options?.code;

  if (!code) {
    if (moduleId == null) {
      throw new Error(`No shader module ID found in pipeline #${pipelineId} ${stage} descriptor`);
    }

    const shaderObj = objects[String(moduleId)] || objects[moduleId];
    if (!shaderObj) {
      throw new Error(
        `Shader module #${moduleId} not found in capture objects (it may have been created before frame capture started). You can pass 'code' to supply the WGSL source directly.`
      );
    }

    code = shaderObj.descriptor?.code || shaderObj.code;
    if (!code || typeof code !== "string" || !code.trim()) {
      throw new Error(`Shader module #${moduleId} contains no WGSL code`);
    }
  }

  let entryPoint = requestedEntryPoint;
  if (!entryPoint) {
    try {
      const reflect = new WgslReflect(code);
      if (stageDesc?.entryPoint && reflect.getFunctionInfo(stageDesc.entryPoint)) {
        entryPoint = stageDesc.entryPoint;
      } else if (reflect.entry[stage]?.length > 0) {
        entryPoint = reflect.entry[stage][0].name;
      }
    } catch {}
    if (!entryPoint) {
      entryPoint = stageDesc?.entryPoint || "main";
    }
  }

  // 5. Reconstruct active bind groups
  const sessionBindGroups = buildSessionBindGroups(capture, passState.bindGroups, payloadResolver);

  // Provide fallback buffers for any unbound storage/uniform buffers in WGSL
  try {
    const reflect = new WgslReflect(code);
    for (const s of reflect.storage || []) {
      if (!sessionBindGroups[s.group]) {
        sessionBindGroups[s.group] = {};
      }
      if (!sessionBindGroups[s.group][s.binding]) {
        const defaultBuf = new ArrayBuffer(65536);
        sessionBindGroups[s.group][s.binding] = {
          storage: defaultBuf,
          buffer: defaultBuf
        };
      }
    }
    for (const u of reflect.uniforms || []) {
      if (!sessionBindGroups[u.group]) {
        sessionBindGroups[u.group] = {};
      }
      if (!sessionBindGroups[u.group][u.binding]) {
        const defaultBuf = new ArrayBuffer(1024);
        sessionBindGroups[u.group][u.binding] = {
          uniform: defaultBuf,
          buffer: defaultBuf
        };
      }
    }
  } catch {}

  // 6. Resolve stage-specific inputs
  let sessionInvocation = { ...invocation };
  const stageConfig = {};

  if (stage === "compute") {
    const drawArgs = passState.command.args || [];
    let threadId = [0, 0, 0];
    if (Array.isArray(invocation)) {
      threadId = [invocation[0] || 0, invocation[1] || 0, invocation[2] || 0];
    } else if (invocation.threadId) {
      threadId = [invocation.threadId[0] || 0, invocation.threadId[1] || 0, invocation.threadId[2] || 0];
    } else if (invocation.globalInvocationId) {
      threadId = [
        invocation.globalInvocationId[0] || 0,
        invocation.globalInvocationId[1] || 0,
        invocation.globalInvocationId[2] || 0
      ];
    } else if (invocation.dispatchId) {
      threadId = [
        invocation.dispatchId[0] || 0,
        invocation.dispatchId[1] || 0,
        invocation.dispatchId[2] || 0
      ];
    } else if (invocation.x !== undefined || invocation.y !== undefined) {
      threadId = [invocation.x || 0, invocation.y || 0, invocation.z || 0];
    }

    const dispatchCount = invocation.dispatchCount || [
      Math.max(1, Number(drawArgs[0]) || 1),
      Math.max(1, Number(drawArgs[1]) || 1),
      Math.max(1, Number(drawArgs[2]) || 1)
    ];

    sessionInvocation = {
      ...sessionInvocation,
      threadId,
      dispatchId: threadId,
      globalInvocationId: threadId,
      dispatchCount
    };
    stageConfig.dispatchCount = dispatchCount;
  } else if (stage === "vertex") {
    const drawArgs = passState.command.args || [];
    const vertexIndex =
      invocation.vertexIndex ?? invocation.vertex_index ?? (Number(drawArgs[2]) || 0);
    const instanceIndex =
      invocation.instanceIndex ?? invocation.instance_index ?? (Number(drawArgs[3]) || 0);

    // Resolve vertex buffer payloads
    const resolvedVertexBuffers = passState.vertexBuffers.map((vb) => {
      let bData = null;
      if (vb.bufferData && vb.bufferData.length > 0) {
        bData = resolvePayloadBytes(vb.bufferData[0], payloadResolver, capture);
      }
      if (!bData && vb.bufferId != null) {
        bData = resolveBufferData(
          vb.bufferId,
          objects[String(vb.bufferId)],
          null,
          capture,
          payloadResolver
        );
      }
      return {
        slot: vb.slot,
        buffer: toArrayBuffer(bData),
        offset: vb.offset || 0,
        size: vb.size
      };
    });

    const vertexInputs = fetchVertexInputs(
      pipeDesc,
      resolvedVertexBuffers,
      vertexIndex,
      instanceIndex,
      invocation.inputs || invocation
    );

    sessionInvocation = {
      vertexIndex,
      instanceIndex,
      ...sessionInvocation,
      ...vertexInputs,
      inputs: vertexInputs
    };
    stageConfig.inputs = vertexInputs;
  } else if (stage === "fragment") {
    // Resolve vertex and index buffers
    const resolvedVertexBuffers = passState.vertexBuffers.map((vb) => {
      let bData = null;
      if (vb.bufferData && vb.bufferData.length > 0) {
        bData = resolvePayloadBytes(vb.bufferData[0], payloadResolver, capture);
      }
      if (!bData && vb.bufferId != null) {
        bData = resolveBufferData(
          vb.bufferId,
          objects[String(vb.bufferId)],
          null,
          capture,
          payloadResolver
        );
      }
      return {
        slot: vb.slot,
        buffer: toArrayBuffer(bData),
        offset: vb.offset || 0,
        size: vb.size
      };
    });

    let resolvedIndexBuffer = null;
    if (passState.indexBuffer) {
      let bData = null;
      if (passState.indexBuffer.bufferData && passState.indexBuffer.bufferData.length > 0) {
        bData = resolvePayloadBytes(passState.indexBuffer.bufferData[0], payloadResolver, capture);
      }
      if (!bData && passState.indexBuffer.bufferId != null) {
        bData = resolveBufferData(
          passState.indexBuffer.bufferId,
          objects[String(passState.indexBuffer.bufferId)],
          null,
          capture,
          payloadResolver
        );
      }
      resolvedIndexBuffer = {
        ...passState.indexBuffer,
        buffer: toArrayBuffer(bData)
      };
    }

    const fragmentInputs = prepareFragmentInputs({
      capture,
      pipeline: pipelineObj,
      drawCmd: passState.command,
      vertexBuffers: resolvedVertexBuffers,
      indexBuffer: resolvedIndexBuffer,
      renderPass: passState.renderPass,
      bindGroups: passState.bindGroups,
      pixelX: invocation.pixelX ?? invocation.x ?? 0,
      pixelY: invocation.pixelY ?? invocation.y ?? 0,
      invocation,
      payloadResolver
    });

    sessionInvocation = {
      ...sessionInvocation,
      ...fragmentInputs,
      inputs: fragmentInputs
    };
    stageConfig.inputs = fragmentInputs;
  }

  // 7. Instantiate and initialize session
  const session = new ShaderDebugSession({
    id: sessionId,
    code,
    stage,
    entryPoint,
    invocation: sessionInvocation,
    bindGroups: sessionBindGroups,
    options: {
      constants: options.constants,
      ...options
    },
    stageConfig
  });

  session.init();
  return session;
}
