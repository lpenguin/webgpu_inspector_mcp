#!/usr/bin/env node
import process from "node:process";
import path from "node:path";
import fs from "node:fs/promises";
import {
  createTestServer,
  createTestClient,
  createSampleCapture
} from "../test/mcp-harness.js";

function printHelp() {
  console.log(`
WebGPU Inspector MCP Call Utility

Usage:
  node bin/mcp-call.js <tool_name> [args_json] [options]
  node bin/mcp-call.js list_tools

Options:
  --help, -h             Show this help message
  --capture, -c <file>   Load a capture file (.wgpuc or .json) before running tool
  --synthetic, -s        Seed store with sample synthetic capture before running tool
  --captures-dir <dir>   Path to captures directory (default: ./captures)
  --raw                  Print raw MCP response object instead of text content

Examples:
  node bin/mcp-call.js list_tools
  node bin/mcp-call.js list_captures
  node bin/mcp-call.js get_capture_summary '{"captureId": "cap-1"}'
  node bin/mcp-call.js get_commands --synthetic method=draw limit=5
  node bin/mcp-call.js get_shader '{"objectId": 1}' --synthetic
  node bin/mcp-call.js load_capture_file '{"path": "captures/cap-1.wgpuc"}'
`);
}

function parseArgValue(str) {
  if (str === "true") return true;
  if (str === "false") return false;
  if (str === "null") return null;
  const num = Number(str);
  if (!Number.isNaN(num) && str.trim() !== "") return num;
  return str;
}

function parseCli(argv) {
  const flags = {
    toolName: null,
    toolArgs: {},
    captureFile: null,
    synthetic: false,
    capturesDir: path.resolve(process.cwd(), "captures"),
    raw: false,
    help: false
  };

  const positional = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      flags.help = true;
    } else if (arg === "--capture" || arg === "-c") {
      flags.captureFile = argv[++i];
    } else if (arg === "--synthetic" || arg === "-s") {
      flags.synthetic = true;
    } else if (arg === "--captures-dir") {
      flags.capturesDir = argv[++i];
    } else if (arg === "--raw") {
      flags.raw = true;
    } else if (arg.startsWith("--capture=")) {
      flags.captureFile = arg.slice("--capture=".length);
    } else if (arg.startsWith("--captures-dir=")) {
      flags.capturesDir = arg.slice("--captures-dir=".length);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length > 0) {
    flags.toolName = positional[0];
  }

  const remaining = positional.slice(1);
  if (remaining.length === 1 && remaining[0].trim().startsWith("{")) {
    try {
      flags.toolArgs = JSON.parse(remaining[0]);
    } catch (err) {
      console.error(`Invalid JSON arguments: ${err.message}`);
      process.exit(1);
    }
  } else if (remaining.length > 0) {
    for (const item of remaining) {
      const eqIdx = item.indexOf("=");
      if (eqIdx !== -1) {
        const key = item.slice(0, eqIdx);
        const val = item.slice(eqIdx + 1);
        flags.toolArgs[key] = parseArgValue(val);
      }
    }
  }

  return flags;
}

async function main() {
  const flags = parseCli(process.argv.slice(2));

  if (flags.help || !flags.toolName) {
    printHelp();
    process.exit(flags.help ? 0 : 1);
  }

  const server = await createTestServer({
    capturesDir: flags.capturesDir
  });
  const harness = await createTestClient(server);

  try {
    if (
      flags.toolName === "list_tools" ||
      flags.toolName === "list-tools" ||
      flags.toolName === "tools"
    ) {
      const tools = await harness.listTools();
      if (flags.raw) {
        console.log(JSON.stringify(tools, null, 2));
      } else {
        console.log(`\nAvailable MCP Tools (${tools.length}):\n`);
        for (const t of tools) {
          console.log(`  • ${t.name.padEnd(24)} ${t.description.split("\n")[0]}`);
        }
        console.log("");
      }
      return;
    }

    if (flags.captureFile) {
      const absPath = path.resolve(process.cwd(), flags.captureFile);
      await harness.store.addFile(absPath);
    } else if (flags.synthetic) {
      const sample = createSampleCapture();
      await harness.loadSyntheticCapture(sample, { label: "synthetic sample" });
    }

    const rawResult = await harness.callTool(flags.toolName, flags.toolArgs);

    if (flags.raw) {
      console.log(JSON.stringify(rawResult, null, 2));
      if (rawResult.isError) {
        process.exit(1);
      }
      return;
    }

    if (rawResult.isError) {
      const errorText = rawResult.content?.[0]?.text || "Tool execution failed";
      console.error(`Error: ${errorText}`);
      process.exit(1);
    }

    const textContent = rawResult.content?.find((c) => c.type === "text");
    if (textContent && typeof textContent.text === "string") {
      try {
        const parsed = JSON.parse(textContent.text);
        console.log(JSON.stringify(parsed, null, 2));
      } catch {
        console.log(textContent.text);
      }
    } else {
      console.log(JSON.stringify(rawResult, null, 2));
    }
  } finally {
    await harness.close();
  }
}

main().catch((err) => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
