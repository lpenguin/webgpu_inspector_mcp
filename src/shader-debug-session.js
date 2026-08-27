import {
  WgslDebug,
  WgslReflect,
  WgslExec
} from "wgsl_reflect/wgsl_reflect.module.js";

/**
 * Decode half-precision float (f16) from uint16.
 */
export function decodeFloat16(u16) {
  const sign = (u16 & 0x8000) >> 15;
  const exp = (u16 & 0x7c00) >> 10;
  const frac = u16 & 0x03ff;
  if (exp === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  } else if (exp === 0x1f) {
    return frac ? NaN : sign ? -Infinity : Infinity;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Decode binary TypedData (buffer with typeInfo) into JS values using DataView for safe alignment.
 */
export function decodeTypedValue(typeInfo, buffer, offset = 0, depth = 0, maxDepth = 4) {
  if (!typeInfo || !buffer || depth > maxDepth) return null;
  const typeName = typeInfo.name || typeInfo.getTypeName?.() || "";
  const buf = buffer instanceof ArrayBuffer ? buffer : (buffer.buffer || buffer);
  const baseOffset = (buffer.byteOffset || 0) + offset;
  if (baseOffset >= buf.byteLength) return null;
  const view = new DataView(buf);

  // Struct
  if (typeInfo.isStruct || typeInfo.members) {
    const result = {};
    for (const m of typeInfo.members) {
      result[m.name] = decodeTypedValue(
        m.type,
        buf,
        baseOffset + (m.offset || 0),
        depth + 1,
        maxDepth
      );
    }
    return result;
  }

  // Array
  if (typeInfo.isArray || typeName === "array") {
    const stride = typeInfo.stride || typeInfo.format?.size || 4;
    const count =
      typeInfo.count > 0
        ? typeInfo.count
        : Math.floor((buf.byteLength - baseOffset) / stride);
    const result = [];
    const safeCount = Math.min(Math.max(0, count), 100);
    for (let i = 0; i < safeCount; i++) {
      result.push(
        decodeTypedValue(
          typeInfo.format,
          buf,
          baseOffset + i * stride,
          depth + 1,
          maxDepth
        )
      );
    }
    return result;
  }

  // Scalars
  if (typeName === "f32") {
    return baseOffset + 4 <= buf.byteLength ? view.getFloat32(baseOffset, true) : null;
  }
  if (typeName === "u32") {
    return baseOffset + 4 <= buf.byteLength ? view.getUint32(baseOffset, true) : null;
  }
  if (typeName === "i32") {
    return baseOffset + 4 <= buf.byteLength ? view.getInt32(baseOffset, true) : null;
  }
  if (typeName === "bool") {
    return baseOffset + 4 <= buf.byteLength ? view.getUint32(baseOffset, true) !== 0 : null;
  }
  if (typeName === "f16") {
    return baseOffset + 2 <= buf.byteLength ? decodeFloat16(view.getUint16(baseOffset, true)) : null;
  }

  // Vectors
  if (typeName.startsWith("vec")) {
    const numComponents =
      parseInt(typeName[3], 10) ||
      (typeName.startsWith("vec2") ? 2 : typeName.startsWith("vec3") ? 3 : 4);
    const formatName =
      typeInfo.format?.name ||
      (typeName.endsWith("f")
        ? "f32"
        : typeName.endsWith("u")
        ? "u32"
        : typeName.endsWith("i")
        ? "i32"
        : typeName.endsWith("h")
        ? "f16"
        : "f32");

    const result = [];
    const compSize = formatName === "f16" ? 2 : 4;
    for (let c = 0; c < numComponents; c++) {
      const o = baseOffset + c * compSize;
      if (o + compSize > buf.byteLength) break;
      if (formatName === "u32") {
        result.push(view.getUint32(o, true));
      } else if (formatName === "i32") {
        result.push(view.getInt32(o, true));
      } else if (formatName === "f16") {
        result.push(decodeFloat16(view.getUint16(o, true)));
      } else {
        result.push(view.getFloat32(o, true));
      }
    }
    return result;
  }

  // Matrices (std140/std430 column stride: 3-row matrices are padded to 4 floats = 16 bytes per column)
  if (typeName.startsWith("mat")) {
    const cols = parseInt(typeName[3], 10) || 2;
    const rows = parseInt(typeName[5], 10) || 2;
    const isF16 = typeName.endsWith("h") || typeInfo.format?.name === "f16";
    const compSize = isF16 ? 2 : 4;
    // In WGSL layout, vec3 columns have 16-byte alignment (or 8-byte in f16)
    const colStride = (rows === 3 ? 4 : rows) * compSize;
    const result = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const o = baseOffset + c * colStride + r * compSize;
        if (o + compSize <= buf.byteLength) {
          result.push(isF16 ? decodeFloat16(view.getUint16(o, true)) : view.getFloat32(o, true));
        } else {
          result.push(0);
        }
      }
    }
    return result;
  }

  return null;
}

/**
 * Formats any WgslDebug / WgslExec Data object into a clean JS representation.
 */
export function formatDataValue(val, maxDepth = 4) {
  if (val === null || val === undefined || maxDepth < 0) return null;
  if (typeof val === "number" || typeof val === "boolean" || typeof val === "string") {
    return val;
  }
  if (Array.isArray(val)) {
    return val.map((v) => formatDataValue(v, maxDepth - 1));
  }

  // PointerData: unwrap reference
  if (val.reference !== undefined || val.constructor?.name === "PointerData" || val.isPointer) {
    if (val.reference !== undefined && val.reference !== val) {
      return formatDataValue(val.reference, maxDepth);
    }
  }

  const typeName = val.typeInfo?.name || val.typeInfo?.getTypeName?.() || "";

  // Textures
  if (typeName.includes("texture") || val.descriptor || val.texture) {
    return {
      type: "texture",
      format: val.typeInfo?.format?.name || typeName || "texture",
      descriptor: val.descriptor || null,
      view: val.view || null
    };
  }

  // Samplers
  if (typeName.includes("sampler") || val.sampler) {
    return {
      type: "sampler",
      descriptor: val.sampler || null
    };
  }

  // ScalarData (.value) - check before .data
  try {
    if (val.value !== undefined) {
      if (val.typeInfo?.name === "bool") {
        return Boolean(val.value);
      }
      return val.value;
    }
  } catch {}

  // TypedData / TrackedTypedData (buffer + typeInfo)
  if (val.buffer && val.typeInfo) {
    return decodeTypedValue(val.typeInfo, val.buffer, val.offset || 0, 0, maxDepth);
  }

  // VectorData / MatrixData with typed array .data
  try {
    if (val.data && val.data.length !== undefined) {
      return Array.from(val.data);
    }
  } catch {}

  // Generic Object
  if (typeof val === "object") {
    if (
      val.constructor?.name === "WgslDebug" ||
      val.constructor?.name === "Context" ||
      val.constructor?.name === "ExecutionState" ||
      val.constructor?.name === "AstNode"
    ) {
      return null;
    }
    const out = {};
    for (const [k, v] of Object.entries(val)) {
      if (
        typeof v !== "function" &&
        k !== "parent" &&
        k !== "context" &&
        k !== "_execStack" &&
        k !== "debugger"
      ) {
        try {
          out[k] = formatDataValue(v, maxDepth - 1);
        } catch {
          out[k] = null;
        }
      }
    }
    return out;
  }
  return val;
}

const SWIZZLE_MAP = {
  x: 0, y: 1, z: 2, w: 3,
  r: 0, g: 1, b: 2, a: 3,
  u: 0, v: 1, s: 0, t: 1
};

/**
 * ShaderDebugSession represents an active interactive shader execution session.
 */
export class ShaderDebugSession {
  constructor(options = {}) {
    this.id =
      options.id ||
      `sdbg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    this.code = options.code || "";
    this.reflect = new WgslReflect(this.code);
    this.stage = options.stage || this._inferStage(options.entryPoint);
    this.entryPoint = options.entryPoint || this._inferEntryPoint();
    this.invocation = options.invocation || {};
    this.bindGroups = options.bindGroups || {};
    this.options = options.options || {};
    this.stageConfig = options.stageConfig || {};
    this.debugger = options.debugger || new WgslDebug(this.code);

    this.status = "uninitialized";
    this.stepCount = 0;
    this.createdAt = Date.now();
    this.lastAccessedAt = Date.now();

    if (options.breakpoints) {
      this.setBreakpoints({ add: Array.from(options.breakpoints) });
    }
  }

  _inferStage(entryPointName) {
    if (entryPointName) {
      const fn = this.reflect.getFunctionInfo(entryPointName);
      if (fn?.stage) return fn.stage;
    }
    if (this.reflect.entry.compute?.length > 0) return "compute";
    if (this.reflect.entry.vertex?.length > 0) return "vertex";
    if (this.reflect.entry.fragment?.length > 0) return "fragment";
    return "compute";
  }

  _inferEntryPoint() {
    if (this.reflect.entry[this.stage]?.length > 0) {
      return this.reflect.entry[this.stage][0].name;
    }
    for (const s of ["compute", "vertex", "fragment"]) {
      if (this.reflect.entry[s]?.length > 0) {
        return this.reflect.entry[s][0].name;
      }
    }
    return "main";
  }

  /**
   * Initializes / starts the debug session for the configured stage and entry point.
   */
  init() {
    this.touch();
    let success = false;
    const config = {
      constants: this.options.constants,
      ...this.stageConfig
    };

    if (this.stage === "compute") {
      let dispatchId = [0, 0, 0];
      if (Array.isArray(this.invocation)) {
        dispatchId = [this.invocation[0] || 0, this.invocation[1] || 0, this.invocation[2] || 0];
      } else if (this.invocation.threadId) {
        dispatchId = [
          this.invocation.threadId[0] || 0,
          this.invocation.threadId[1] || 0,
          this.invocation.threadId[2] || 0
        ];
      } else if (this.invocation.globalInvocationId) {
        dispatchId = [
          this.invocation.globalInvocationId[0] || 0,
          this.invocation.globalInvocationId[1] || 0,
          this.invocation.globalInvocationId[2] || 0
        ];
      } else if (this.invocation.dispatchId) {
        dispatchId = [
          this.invocation.dispatchId[0] || 0,
          this.invocation.dispatchId[1] || 0,
          this.invocation.dispatchId[2] || 0
        ];
      }

      let dispatchCount =
        this.invocation.dispatchCount ||
        this.stageConfig.dispatchCount || [
          Math.max(1, dispatchId[0] + 1),
          Math.max(1, dispatchId[1] + 1),
          Math.max(1, dispatchId[2] + 1)
        ];

      success = this.debugger.debugWorkgroup(
        this.entryPoint,
        dispatchId,
        dispatchCount,
        this.bindGroups,
        config
      );
    } else if (this.stage === "vertex") {
      const stageInputs = this._normalizeVertexInputs(this.invocation, this.stageConfig);
      success = this.debugger.debugVertex(
        this.entryPoint,
        stageInputs,
        this.bindGroups,
        config
      );
    } else if (this.stage === "fragment") {
      const stageInputs = this._normalizeFragmentInputs(this.invocation, this.stageConfig);
      success = this.debugger.debugFragment(
        this.entryPoint,
        stageInputs,
        this.bindGroups,
        config
      );
    }

    if (!success) {
      this.status = "error";
      throw new Error(
        `Failed to initialize debug session for stage "${this.stage}", entry "${this.entryPoint}"`
      );
    }

    this.status = this.isAtEnd ? "completed" : "paused";
    return this.getStateSnapshot();
  }

  _normalizeVertexInputs(invocation, stageConfig) {
    const raw = {
      ...(stageConfig?.inputs || {}),
      ...(invocation?.inputs || {}),
      ...invocation
    };
    const inputs = { ...raw };

    const fnInfo = this.reflect.getFunctionInfo(this.entryPoint);
    if (fnInfo?.inputs) {
      for (const inp of fnInfo.inputs) {
        if (raw[inp.name] !== undefined) {
          inputs[inp.location] = raw[inp.name];
          inputs[String(inp.location)] = raw[inp.name];
        }
      }
    }

    if (raw.vertexIndex !== undefined) {
      inputs.vertex_index = raw.vertexIndex;
    }
    if (raw.instanceIndex !== undefined) {
      inputs.instance_index = raw.instanceIndex;
    }
    if (inputs.vertex_index === undefined) inputs.vertex_index = 0;
    if (inputs.instance_index === undefined) inputs.instance_index = 0;

    return inputs;
  }

  _normalizeFragmentInputs(invocation, stageConfig) {
    const raw = {
      ...(stageConfig?.inputs || {}),
      ...(invocation?.inputs || {}),
      ...invocation
    };
    const inputs = { ...raw };

    const fnInfo = this.reflect.getFunctionInfo(this.entryPoint);
    if (fnInfo?.inputs) {
      for (const inp of fnInfo.inputs) {
        if (raw[inp.name] !== undefined) {
          inputs[inp.location] = raw[inp.name];
          inputs[String(inp.location)] = raw[inp.name];
        }
      }
    }

    if (raw.sampleIndex !== undefined) {
      inputs.sample_index = raw.sampleIndex;
    }
    if (raw.sampleMask !== undefined) {
      inputs.sample_mask = raw.sampleMask;
    }
    if (raw.frontFacing !== undefined) {
      inputs.front_facing = raw.frontFacing;
    }

    if (inputs.position === undefined && (raw.x !== undefined || raw.y !== undefined)) {
      const x = raw.x !== undefined ? raw.x + 0.5 : 0.5;
      const y = raw.y !== undefined ? raw.y + 0.5 : 0.5;
      const z = raw.z !== undefined ? raw.z : 0.5;
      const w = raw.w !== undefined ? raw.w : 1.0;
      inputs.position = [x, y, z, w];
    }
    return inputs;
  }

  get isAtEnd() {
    return !this.debugger.currentState || this.debugger.currentState.isAtEnd;
  }

  /**
   * Steps the execution.
   * @param {"step_over" | "step_into" | "step_out" | "step_next"} action
   * @param {number} count Number of steps to execute
   */
  step(action = "step_next", count = 1) {
    this.touch();
    if (this.status === "uninitialized") {
      this.init();
    }
    if (this.isAtEnd) {
      this.status = "completed";
      return this.getStateSnapshot();
    }

    const stepsToRun = Math.max(1, count || 1);
    const subStepLimit = 10000;

    for (let c = 0; c < stepsToRun; c++) {
      if (this.isAtEnd) {
        this.status = "completed";
        break;
      }

      const startLine = this.debugger.currentLine;
      const startDepth = this.debugger._execStack?.states?.length || 0;
      let innerSteps = 0;

      if (action === "step_into") {
        this.debugger.stepNext(true);
        this.stepCount++;
        while (
          !this.isAtEnd &&
          this.debugger.currentLine === startLine &&
          (this.debugger._execStack?.states?.length || 0) === startDepth &&
          innerSteps++ < subStepLimit
        ) {
          if (this.debugger.breakpoints.has(this.debugger.currentLine)) break;
          this.debugger.stepNext(true);
          this.stepCount++;
        }
      } else if (action === "step_over") {
        this.debugger.stepNext(false);
        this.stepCount++;
        while (
          !this.isAtEnd &&
          this.debugger.currentLine === startLine &&
          (this.debugger._execStack?.states?.length || 0) === startDepth &&
          innerSteps++ < subStepLimit
        ) {
          if (this.debugger.breakpoints.has(this.debugger.currentLine)) break;
          this.debugger.stepNext(false);
          this.stepCount++;
        }
      } else if (action === "step_next") {
        this.debugger.stepNext(true);
        this.stepCount++;
        while (
          !this.isAtEnd &&
          this.debugger.currentLine === startLine &&
          (this.debugger._execStack?.states?.length || 0) === startDepth &&
          innerSteps++ < subStepLimit
        ) {
          if (this.debugger.breakpoints.has(this.debugger.currentLine)) break;
          this.debugger.stepNext(true);
          this.stepCount++;
        }
      } else if (action === "step_out") {
        let fnRoot = this.debugger.currentState;
        while (fnRoot && fnRoot.parent && !fnRoot.parentCallExpr) {
          fnRoot = fnRoot.parent;
        }
        const callerFrame = fnRoot ? fnRoot.parent : null;
        let outSteps = 0;
        while (this.debugger.currentState && outSteps < subStepLimit) {
          const more = this.debugger.stepNext(true);
          outSteps++;
          this.stepCount++;
          if (!more || !this.debugger.currentState) break;
          if (this.debugger.breakpoints.has(this.debugger.currentLine)) break;
          if (
            this.debugger.currentState === callerFrame ||
            !this.debugger._execStack?.states.includes(fnRoot)
          ) {
            break;
          }
        }
      }

      if (this.debugger.breakpoints.has(this.debugger.currentLine)) {
        this.status = "paused";
        break;
      }
      if (this.isAtEnd) {
        this.status = "completed";
        break;
      }
    }

    if (this.isAtEnd) {
      this.status = "completed";
    } else {
      this.status = "paused";
    }

    return this.getStateSnapshot();
  }

  /**
   * Continues execution until a breakpoint is hit, shader completes, or maxSteps exceeded.
   * @param {number} maxSteps
   */
  continueExecution(maxSteps = 50000) {
    this.touch();
    if (this.status === "uninitialized") {
      this.init();
    }
    if (this.isAtEnd) {
      this.status = "completed";
      return this.getStateSnapshot();
    }

    let steps = 0;
    let hitBreakpoint = null;

    const startLine = this.debugger.currentLine;
    const firstMore = this.debugger.stepNext(true);
    steps++;
    this.stepCount++;

    if (!firstMore || this.isAtEnd) {
      this.status = "completed";
    } else if (
      this.debugger.currentLine !== startLine &&
      this.debugger.breakpoints.has(this.debugger.currentLine)
    ) {
      hitBreakpoint = this.debugger.currentLine;
      this.status = "paused";
    }

    while (!hitBreakpoint && !this.isAtEnd && steps < maxSteps) {
      const more = this.debugger.stepNext(true);
      steps++;
      this.stepCount++;
      if (!more || this.isAtEnd) {
        this.status = "completed";
        break;
      }
      if (this.debugger.breakpoints.has(this.debugger.currentLine)) {
        hitBreakpoint = this.debugger.currentLine;
        this.status = "paused";
        break;
      }
    }

    if (this.isAtEnd) {
      this.status = "completed";
    } else {
      this.status = "paused";
    }

    const snapshot = this.getStateSnapshot();
    if (hitBreakpoint !== null) {
      snapshot.hitBreakpoint = hitBreakpoint;
    }
    return snapshot;
  }

  /**
   * Sets/removes breakpoints on the active debugger.
   * @param {{ add?: number[], remove?: number[], clearAll?: boolean }} options
   */
  setBreakpoints({ add = [], remove = [], clearAll = false } = {}) {
    this.touch();
    if (clearAll) {
      this.debugger.clearBreakpoints();
    }
    if (Array.isArray(remove)) {
      for (const l of remove) {
        this.debugger.breakpoints.delete(Number(l));
      }
    }
    if (Array.isArray(add)) {
      for (const l of add) {
        this.debugger.breakpoints.add(Number(l));
      }
    }
    return Array.from(this.debugger.breakpoints).sort((a, b) => a - b);
  }

  /**
   * Returns function-level callstack [{ level, functionName, line }], innermost frame first.
   */
  getCallstack() {
    if (!this.debugger || !this.debugger._execStack) return [];
    const states = this.debugger._execStack.states;
    if (!states || states.length === 0) return [];

    const frames = [];
    let currentGroup = [];

    for (let i = 0; i < states.length; i++) {
      const st = states[i];
      if (i > 0 && st.parentCallExpr) {
        if (currentGroup.length > 0) {
          frames.push(currentGroup);
        }
        currentGroup = [st];
      } else {
        currentGroup.push(st);
      }
    }
    if (currentGroup.length > 0) {
      frames.push(currentGroup);
    }

    const result = [];
    for (let f = 0; f < frames.length; f++) {
      const group = frames[f];
      const topState = group[group.length - 1];
      const rootState = group[0];
      const functionName =
        topState.context?.currentFunctionName ||
        rootState.context?.currentFunctionName ||
        (f === 0 ? this.entryPoint : "anonymous");

      let line = -1;
      for (let s = group.length - 1; s >= 0; s--) {
        const cmd = group[s].getCurrentCommand();
        if (cmd && cmd.line && cmd.line > 0) {
          line = cmd.line;
          break;
        }
      }
      if (line <= 0 && f === frames.length - 1) {
        line = this.debugger.currentLine;
      }
      if (line <= 0 && rootState.parentCallExpr?.line) {
        line = rootState.parentCallExpr.line;
      }

      result.push({ functionName, line });
    }

    return result.reverse().map((frame, index) => ({
      level: index,
      functionName: frame.functionName,
      line: frame.line
    }));
  }

  /**
   * Returns variables organized by category (locals, inputs, globals, constants, resources).
   */
  getVariables({ scope = "all", filter = "", maxDepth = 2 } = {}) {
    this.touch();
    const locals = {};
    const inputs = {};
    const globals = {};
    const constants = {};
    const resources = {};

    const activeContext = this.debugger.context;
    const globalContext = this.debugger.exec?.context;

    // Identify uniform / storage / texture / sampler names from reflection
    const uniformNames = new Set((this.reflect.uniforms || []).map((u) => u.name));
    const storageNames = new Set((this.reflect.storage || []).map((s) => s.name));
    const textureNames = new Set((this.reflect.textures || []).map((t) => t.name));
    const samplerNames = new Set((this.reflect.samplers || []).map((s) => s.name));
    const overrideNames = new Set((this.reflect.overrides || []).map((o) => o.name));

    // Entry point argument names
    const entryFuncInfo = this.reflect.getFunctionInfo(this.entryPoint);
    const entryArgNames = new Set([
      ...(entryFuncInfo?.arguments || []).map((arg) => arg.name),
      ...(entryFuncInfo?.inputs || []).map((inp) => inp.name)
    ]);

    const isInsideHelper =
      activeContext?.currentFunctionName &&
      activeContext.currentFunctionName !== this.entryPoint;

    // Walk up the scope parent chain from activeContext to collect local variables
    const localVarsMap = new Map();
    let ctx = activeContext;
    while (ctx && ctx !== globalContext) {
      if (ctx.variables) {
        for (const [k, v] of ctx.variables.entries()) {
          if (!localVarsMap.has(k)) {
            localVarsMap.set(k, v);
          }
        }
      }
      ctx = ctx.parent;
    }

    // Process local variables
    for (const [name, varRef] of localVarsMap.entries()) {
      const val = varRef?.value;
      const formatted = formatDataValue(val, maxDepth);
      if (name.startsWith("@")) {
        inputs[name] = formatted;
      } else if (entryArgNames.has(name) && !isInsideHelper) {
        inputs[name] = formatted;
      } else if (uniformNames.has(name) || storageNames.has(name)) {
        globals[name] = formatted;
      } else if (overrideNames.has(name) || varRef?.node?.constructor?.name === "Const") {
        constants[name] = formatted;
      } else {
        locals[name] = formatted;
      }
    }

    // Process global context variables (globals, consts, resources, and entrypoint inputs if at root)
    if (globalContext?.variables) {
      for (const [name, varRef] of globalContext.variables.entries()) {
        const val = varRef?.value;
        const formatted = formatDataValue(val, maxDepth);

        if (textureNames.has(name)) {
          const texInfo = (this.reflect.textures || []).find((t) => t.name === name);
          const bound = this.bindGroups[texInfo?.group]?.[texInfo?.binding];
          if (bound?.descriptor && typeof formatted === "object" && formatted !== null) {
            formatted.descriptor = bound.descriptor;
          }
          resources[name] = formatted;
        } else if (samplerNames.has(name)) {
          const sampInfo = (this.reflect.samplers || []).find((s) => s.name === name);
          const bound = this.bindGroups[sampInfo?.group]?.[sampInfo?.binding];
          if (bound?.sampler && typeof formatted === "object" && formatted !== null) {
            formatted.descriptor = bound.sampler;
          }
          resources[name] = formatted;
        } else if (name.startsWith("@")) {
          inputs[name] = formatted;
        } else if (entryArgNames.has(name)) {
          inputs[name] = formatted;
        } else if (uniformNames.has(name) || storageNames.has(name)) {
          globals[name] = formatted;
        } else if (overrideNames.has(name) || varRef?.node?.constructor?.name === "Const") {
          constants[name] = formatted;
        } else if (!isInsideHelper && !(name in locals)) {
          locals[name] = formatted;
        }
      }
    }

    // Include reflections textures & samplers if not already populated
    for (const tex of this.reflect.textures || []) {
      if (!(tex.name in resources)) {
        const bound = this.bindGroups[tex.group]?.[tex.binding];
        resources[tex.name] = {
          type: "texture",
          format: tex.type?.format?.name || tex.type?.name || "texture",
          descriptor: bound?.descriptor || null,
          view: bound?.view || null
        };
      }
    }

    for (const samp of this.reflect.samplers || []) {
      if (!(samp.name in resources)) {
        const bound = this.bindGroups[samp.group]?.[samp.binding];
        resources[samp.name] = {
          type: "sampler",
          descriptor: bound?.sampler || null
        };
      }
    }

    // Include reflections constants & overrides not in context
    for (const ov of this.reflect.overrides || []) {
      if (!(ov.name in constants)) {
        constants[ov.name] = this.options.constants?.[ov.name] ?? null;
      }
    }

    // Include invocation inputs if not already present
    if (this.invocation?.inputs && typeof this.invocation.inputs === "object") {
      for (const [k, v] of Object.entries(this.invocation.inputs)) {
        if (!(k in inputs)) {
          inputs[k] = formatDataValue(v, maxDepth);
        }
      }
    }

    const applyFilter = (obj) => {
      if (!filter) return obj;
      const f = filter.toLowerCase();
      const filtered = {};
      for (const [k, v] of Object.entries(obj)) {
        if (k.toLowerCase().includes(f)) {
          filtered[k] = v;
        }
      }
      return filtered;
    };

    const out = {
      locals: applyFilter(locals),
      inputs: applyFilter(inputs),
      globals: applyFilter(globals),
      constants: applyFilter(constants),
      resources: applyFilter(resources)
    };

    if (scope !== "all" && scope in out) {
      return { [scope]: out[scope] };
    }
    return out;
  }

  /**
   * Evaluates a path expression like pos.x, material.color[0], vNormal.xyz in the current context.
   */
  evaluate(path) {
    this.touch();
    if (!path || typeof path !== "string") {
      return { success: false, path, error: "Invalid path expression" };
    }
    const trimmed = path.trim();
    if (!trimmed) {
      return { success: false, path, error: "Empty path expression" };
    }

    const tokenRegex = /([a-zA-Z_@][a-zA-Z0-9_]*)|(\.[a-zA-Z_][a-zA-Z0-9_]*)|(\[\s*\d+\s*\])/g;
    const tokens = [];
    let match;
    let lastIndex = 0;
    while ((match = tokenRegex.exec(trimmed)) !== null) {
      if (match.index !== lastIndex) {
        return {
          success: false,
          path,
          error: `Syntax error near "${trimmed.substring(lastIndex, match.index)}"`
        };
      }
      lastIndex = tokenRegex.lastIndex;
      tokens.push(match[0]);
    }
    if (lastIndex !== trimmed.length) {
      return {
        success: false,
        path,
        error: `Syntax error near "${trimmed.substring(lastIndex)}"`
      };
    }
    if (tokens.length === 0) {
      return { success: false, path, error: "No identifier found in path" };
    }

    const rootName = tokens[0];
    const varRef =
      this.debugger.context?.getVariable(rootName) ||
      this.debugger.exec?.context?.getVariable(rootName);

    if (!varRef) {
      return {
        success: false,
        path,
        error: `Variable "${rootName}" not found in current scope`
      };
    }

    let currentVal = varRef.value;

    for (let i = 1; i < tokens.length; i++) {
      const token = tokens[i];

      // Unwrap PointerData if needed
      if (currentVal && (currentVal.reference !== undefined || currentVal.constructor?.name === "PointerData")) {
        currentVal = currentVal.reference;
      }

      if (token.startsWith(".")) {
        const memberName = token.substring(1);

        // Vector swizzle on .data (VectorData / MatrixData)
        if (currentVal && currentVal.data && currentVal.data.length !== undefined) {
          const vecLen = currentVal.data.length;
          const chars = memberName.split("");
          if (chars.every((c) => c in SWIZZLE_MAP)) {
            for (const c of chars) {
              if (SWIZZLE_MAP[c] >= vecLen) {
                return {
                  success: false,
                  path,
                  error: `Swizzle component '${c}' out of bounds for vector of length ${vecLen}`
                };
              }
            }
            if (chars.length === 1) {
              currentVal = currentVal.data[SWIZZLE_MAP[chars[0]]];
            } else {
              currentVal = chars.map((c) => currentVal.data[SWIZZLE_MAP[c]]);
            }
            continue;
          }
        }

        // Vector swizzle on plain JS array (e.g. from previous swizzle)
        if (Array.isArray(currentVal)) {
          const vecLen = currentVal.length;
          const chars = memberName.split("");
          if (chars.every((c) => c in SWIZZLE_MAP)) {
            for (const c of chars) {
              if (SWIZZLE_MAP[c] >= vecLen) {
                return {
                  success: false,
                  path,
                  error: `Swizzle component '${c}' out of bounds for array of length ${vecLen}`
                };
              }
            }
            if (chars.length === 1) {
              currentVal = currentVal[SWIZZLE_MAP[chars[0]]];
            } else {
              currentVal = chars.map((c) => currentVal[SWIZZLE_MAP[c]]);
            }
            continue;
          }
        }

        // TypedData vector swizzle
        if (
          currentVal &&
          currentVal.typeInfo &&
          currentVal.typeInfo.name?.startsWith("vec") &&
          currentVal.buffer
        ) {
          const vecArr = decodeTypedValue(currentVal.typeInfo, currentVal.buffer, currentVal.offset || 0);
          if (Array.isArray(vecArr)) {
            const vecLen = vecArr.length;
            const chars = memberName.split("");
            if (chars.every((c) => c in SWIZZLE_MAP)) {
              for (const c of chars) {
                if (SWIZZLE_MAP[c] >= vecLen) {
                  return {
                    success: false,
                    path,
                    error: `Swizzle component '${c}' out of bounds for vector of length ${vecLen}`
                  };
                }
              }
              if (chars.length === 1) {
                currentVal = vecArr[SWIZZLE_MAP[chars[0]]];
              } else {
                currentVal = chars.map((c) => vecArr[SWIZZLE_MAP[c]]);
              }
              continue;
            }
          }
        }

        // TypedData struct
        if (
          currentVal &&
          currentVal.typeInfo &&
          (currentVal.typeInfo.isStruct || currentVal.typeInfo.members)
        ) {
          const mem = currentVal.typeInfo.members.find((m) => m.name === memberName);
          if (!mem) {
            return {
              success: false,
              path,
              error: `Member "${memberName}" not found on struct ${currentVal.typeInfo.name}`
            };
          }
          currentVal = {
            typeInfo: mem.type,
            buffer: currentVal.buffer,
            offset: (currentVal.offset || 0) + (mem.offset || 0),
            isTypedData: true
          };
          continue;
        }

        // Decoded / plain JS object
        if (
          typeof currentVal === "object" &&
          currentVal !== null &&
          memberName in currentVal
        ) {
          currentVal = currentVal[memberName];
          continue;
        }

        return {
          success: false,
          path,
          error: `Cannot access member "${memberName}" on value`
        };
      } else if (token.startsWith("[")) {
        const idx = parseInt(token.replace(/\[\s*|\s*\]/g, ""), 10);

        // Matrix Data indexing: m[col] returns column vector of rows
        if (currentVal && currentVal.typeInfo && currentVal.typeInfo.name?.startsWith("mat")) {
          const typeName = currentVal.typeInfo.name;
          const cols = parseInt(typeName[3], 10) || 2;
          const rows = parseInt(typeName[5], 10) || 2;
          if (idx < 0 || idx >= cols) {
            return {
              success: false,
              path,
              error: `Column index ${idx} out of bounds (matrix has ${cols} columns)`
            };
          }
          if (currentVal.data) {
            const colVals = [];
            for (let r = 0; r < rows; r++) {
              colVals.push(currentVal.data[idx * rows + r]);
            }
            currentVal = colVals;
            continue;
          }
          if (currentVal.buffer) {
            const isF16 = typeName.endsWith("h");
            const compSize = isF16 ? 2 : 4;
            const colStride = (rows === 3 ? 4 : rows) * compSize;
            currentVal = {
              typeInfo: { name: `vec${rows}${isF16 ? "h" : "f"}`, format: { name: isF16 ? "f16" : "f32" } },
              buffer: currentVal.buffer,
              offset: (currentVal.offset || 0) + idx * colStride,
              isTypedData: true
            };
            continue;
          }
        }

        // Vector / Typed Array indexing
        if (currentVal && currentVal.data && currentVal.data.length !== undefined) {
          if (idx < 0 || idx >= currentVal.data.length) {
            return {
              success: false,
              path,
              error: `Index ${idx} out of bounds (length ${currentVal.data.length})`
            };
          }
          currentVal = currentVal.data[idx];
          continue;
        }

        // TypedData Array indexing
        if (
          currentVal &&
          currentVal.typeInfo &&
          (currentVal.typeInfo.isArray || currentVal.typeInfo.name === "array")
        ) {
          const stride =
            currentVal.typeInfo.stride || currentVal.typeInfo.format?.size || 4;
          const totalCount =
            currentVal.typeInfo.count > 0
              ? currentVal.typeInfo.count
              : Math.floor(
                  (currentVal.buffer.byteLength - (currentVal.offset || 0)) / stride
                );
          if (idx < 0 || idx >= totalCount) {
            return {
              success: false,
              path,
              error: `Index ${idx} out of bounds (count ${totalCount})`
            };
          }
          currentVal = {
            typeInfo: currentVal.typeInfo.format,
            buffer: currentVal.buffer,
            offset: (currentVal.offset || 0) + idx * stride,
            isTypedData: true
          };
          continue;
        }

        // TypedData Vector indexing
        if (
          currentVal &&
          currentVal.typeInfo &&
          currentVal.typeInfo.name?.startsWith("vec")
        ) {
          const numComponents = parseInt(currentVal.typeInfo.name[3], 10) || 4;
          const compType = currentVal.typeInfo.format || { size: 4, name: "f32" };
          const stride = compType.size || 4;
          if (idx < 0 || idx >= numComponents) {
            return {
              success: false,
              path,
              error: `Index ${idx} out of bounds (vec size ${numComponents})`
            };
          }
          currentVal = {
            typeInfo: compType,
            buffer: currentVal.buffer,
            offset: (currentVal.offset || 0) + idx * stride,
            isTypedData: true
          };
          continue;
        }

        // JS Array indexing
        if (Array.isArray(currentVal)) {
          if (idx < 0 || idx >= currentVal.length) {
            return {
              success: false,
              path,
              error: `Index ${idx} out of bounds (length ${currentVal.length})`
            };
          }
          currentVal = currentVal[idx];
          continue;
        }

        return {
          success: false,
          path,
          error: `Cannot index non-array value`
        };
      }
    }

    const formattedValue = formatDataValue(currentVal);
    return {
      success: true,
      path,
      value: formattedValue,
      type: currentVal?.typeInfo?.name || typeof formattedValue
    };
  }

  /**
   * Returns surrounding lines with line numbers and a `->` marker on `line`.
   */
  getSourceSnippet(line, radius = 2) {
    if (!this.code || line <= 0) return "";
    const lines = this.code.split(/\r?\n/);
    const startLine = Math.max(1, line - radius);
    const endLine = Math.min(lines.length, line + radius);
    const maxLineNumWidth = String(endLine).length;

    const snippetLines = [];
    for (let l = startLine; l <= endLine; l++) {
      const lineText = lines[l - 1] || "";
      const isCurrent = l === line;
      const marker = isCurrent ? "->" : "  ";
      const lineNumStr = String(l).padStart(maxLineNumWidth, " ");
      snippetLines.push(`${marker} ${lineNumStr} | ${lineText}`);
    }
    return snippetLines.join("\n");
  }

  /**
   * Returns complete snapshot of the debug session state.
   */
  getStateSnapshot() {
    this.touch();
    const currentLine = this.debugger.currentLine;
    const currentCommand = this.debugger.currentCommand;
    const activeContext = this.debugger.context;
    const currentFunction =
      activeContext?.currentFunctionName || this.entryPoint || "<shader>";

    const callstack = this.getCallstack();

    return {
      sessionId: this.id,
      stage: this.stage,
      entryPoint: this.entryPoint,
      invocation: this.invocation,
      status: this.status,
      currentLine: currentLine > 0 ? currentLine : null,
      currentFunction,
      currentCommand: currentCommand
        ? {
            line: currentCommand.line,
            type: currentCommand.constructor?.name || "Command"
          }
        : null,
      sourceSnippet: currentLine > 0 ? this.getSourceSnippet(currentLine, 2) : "",
      callstackDepth: callstack.length,
      activeBreakpoints: Array.from(this.debugger.breakpoints).sort((a, b) => a - b),
      isAtEnd: this.isAtEnd,
      stepCount: this.stepCount,
      returnValue: formatDataValue(this.debugger.returnValue),
      discarded: Boolean(this.debugger.discarded)
    };
  }

  touch() {
    this.lastAccessedAt = Date.now();
  }

  dispose() {
    try {
      this.debugger.reset();
    } catch {}
    this.status = "disposed";
  }
}

/**
 * Manages active ShaderDebugSession instances with automatic TTL eviction.
 */
export class ShaderDebugSessionManager {
  constructor(options = {}) {
    this.sessions = new Map();
    this.defaultTtlMs = options.defaultTtlMs || 15 * 60 * 1000;
  }

  createSession(options) {
    this.cleanupExpired();
    const session = new ShaderDebugSession(options);
    session.init();
    this.sessions.set(session.id, session);
    return session;
  }

  getSession(sessionId, maxAgeMs) {
    this.cleanupExpired(maxAgeMs);
    const session = this.sessions.get(sessionId);
    if (session) {
      const ttl = maxAgeMs || this.defaultTtlMs;
      if (Date.now() - session.lastAccessedAt > ttl) {
        session.dispose();
        this.sessions.delete(sessionId);
        return null;
      }
      session.touch();
      return session;
    }
    return null;
  }

  deleteSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.dispose();
      this.sessions.delete(sessionId);
      return true;
    }
    return false;
  }

  listSessions() {
    this.cleanupExpired();
    const list = [];
    for (const [id, session] of this.sessions.entries()) {
      list.push({
        id,
        sessionId: id,
        stage: session.stage,
        entryPoint: session.entryPoint,
        status: session.status,
        currentLine: session.debugger.currentLine,
        createdAt: session.createdAt,
        lastAccessedAt: session.lastAccessedAt
      });
    }
    return list;
  }

  cleanupExpired(maxAgeMs) {
    const ttl = maxAgeMs || this.defaultTtlMs;
    const now = Date.now();
    let count = 0;
    for (const [id, session] of this.sessions.entries()) {
      if (now - session.lastAccessedAt > ttl) {
        session.dispose();
        this.sessions.delete(id);
        count++;
      }
    }
    return count;
  }

  disposeAll() {
    for (const [id, session] of this.sessions.entries()) {
      session.dispose();
    }
    this.sessions.clear();
  }
}
