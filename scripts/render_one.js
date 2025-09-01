// Render a single changed .mmd diagram to SVG, then SVGO optimize.
// Used by: npm run diag:watch (via chokidar-cli)
// Env from chokidar-cli: CHOKIDAR_CLI_PATH, CHOKIDAR_CLI_EVENT
import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

const inFile = process.env.CHOKIDAR_CLI_PATH;
if (!inFile || !inFile.endsWith(".mmd")) {
  process.exit(0);
}
const outFile = inFile.replace(/^diagrams\//, "outputs/diagrams/").replace(/\.mmd$/, ".svg");
mkdirSync(dirname(outFile), { recursive: true });

const cmd = [
  `npx mmdc -i "${inFile}" -o "${outFile}" -b transparent -t neutral`,
  `npx svgo "${outFile}" --config .svgo.json`
].join(" && ");

execSync(cmd, { stdio: "inherit" });
