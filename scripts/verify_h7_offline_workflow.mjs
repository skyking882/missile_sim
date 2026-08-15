import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";


const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const workspaceRoot = path.resolve(projectRoot, "..");
const rawRoot = path.join(projectRoot, "data", "raw", "statshark_h7_controller_id");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function fileSha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

const configPath = path.join(projectRoot, "configs", "h7_controller_experiments.json");
const ledgerPath = path.join(rawRoot, "calculate_ledger.json");
const manifestPath = path.join(rawRoot, "session_manifest.json");
const planPath = path.join(workspaceRoot, "plan7.md");

const config = readJson(configPath);
const ledger = readJson(ledgerPath);
const manifest = readJson(manifestPath);
const planText = fs.readFileSync(planPath, "utf8");
const actions = config.calculate_actions || [];
const r3 = actions.find((entry) => entry.case_id === "H7_R3_DYNAMIC_REVERSAL");
const r3n = actions.find((entry) => entry.case_id === "H7_R3N_NEAR_ENVELOPE_RETRY");
const r3nAction = (ledger.actions || []).find((entry) => entry.case_id === "H7_R3N_NEAR_ENVELOPE_RETRY");
const postR3N = Boolean(r3nAction);

const checks = [];
function check(name, pass, details = null) {
  checks.push({ name, pass: Boolean(pass), details });
}

const preExecutionPolicy = "offline_rules_and_verification_only_no_calculate";
const postExecutionPolicy = "single_explicit_r3n_calculate_consumed_then_pause";
const policy = postR3N ? postExecutionPolicy : preExecutionPolicy;
check("config authorization state is coherent", config.authorization_policy === policy);
check("ledger authorization state is coherent", ledger.authorization_policy === (postR3N ? "explicit_single_r3n_calculate_consumed_then_pause" : policy));
check("manifest authorization state is coherent", manifest.authorization_policy === policy);
check(
  "all current server-action authorizations are zero",
  config.calculate_budget?.current_authorized_server_actions === 0 &&
    ledger.current_authorized_server_actions === 0 &&
    manifest.current_authorized_server_actions === 0,
);
check("ledger action count matches pre/post execution state", ledger.actions?.length === (postR3N ? 6 : 5));
check(
  "R4 has no submission and R3N has at most one submission",
  !(ledger.actions || []).some((entry) => entry.case_id === "H7_R4_FROZEN_HOLDOUT") &&
    (ledger.actions || []).filter((entry) => entry.case_id === "H7_R3N_NEAR_ENVELOPE_RETRY").length <= 1,
);
check("R3 remains excitation-gate failed", r3?.result?.excitation_gate_pass === false);
check(
  "R3N state is design-only before execution or stopped after one executed action",
  postR3N
    ? r3n?.authorization_status === "executed_once_authorized_then_stopped" && r3nAction?.r3n_gate_pass === false
    : r3n?.authorization_status === "design_only_not_authorized",
);
check(
  "R3N permits at most one future action and none now",
  r3n?.maximum_calculate_actions_if_later_authorized === 1 &&
    r3n?.execution_contract?.current_authorized_calculate_actions === 0,
);

const expectedModels = [
  ["H7_R3N_NOM_F0020", 0.845158],
  ["H7_R3N_NOM_F0025", 1.0564475],
  ["H7_R3N_NOM_F100", 42.2579],
];
check(
  "R3N has only the three reviewed authority arms",
  r3n?.models?.length === expectedModels.length &&
    expectedModels.every(([name, authority], index) =>
      r3n.models[index]?.name === name && r3n.models[index]?.finsLatAccel === authority),
);

const sharedScenarioKeys = [
  "StartSpeed",
  "LaunchAltitude",
  "ClosureRate",
  "InitialTargetDistance",
  "TargetAltitude",
  "TargetAzimuth",
  "TargetCourse",
  "TargetConstantGTurn",
];
check(
  "R3N preserves the reviewed R3 geometry",
  sharedScenarioKeys.every((key) => r3n?.scenario?.[key] === r3?.scenario?.[key]),
);
check(
  "R3N freezes controller and allows only finsLatAccel across arms",
  r3n?.execution_contract?.nominal_pid_and_schedule_frozen === true &&
    r3n?.execution_contract?.only_cross_arm_model_field === "finsLatAccel",
);
check(
  "R3N has a one-second strict coverage gate",
  r3n?.acceptance_gate?.at_least_one_low_authority_strict_near_envelope_duration_s === 1.0 &&
    r3n?.acceptance_gate?.F100_unsaturated_duration_s === 1.0,
);
check(
  "post-execution R3N result is recorded without fit or R4",
  !postR3N || (
    r3n?.result?.http_status === 200 &&
    r3n?.result?.schema_pass === true &&
    r3n?.result?.payload_contrast_pass === true &&
    r3n?.result?.excitation_gate_pass === false &&
    r3nAction?.valid_for_controller_fit === false &&
    manifest.r4_cleared === false
  ),
);
check("R4 remains blocked", manifest.r4_cleared === false && ledger.r4_cleared === false);
check(
  "Plan states the zero-Calculate pause boundary",
  planText.includes("当前新增 server Calculate 授权数 = 0") &&
    planText.includes("H7_R3N_NEAR_ENVELOPE_RETRY") &&
    planText.includes("完成离线工作流后必须暂停"),
);

const manifestPlanSource = manifest.sources?.[planPath];
const manifestConfigSource = manifest.sources?.[configPath];
check(
  "manifest plan hash matches current plan",
  manifestPlanSource?.sha256 === fileSha256(planPath),
  { recorded: manifestPlanSource?.sha256, computed: fileSha256(planPath) },
);
check(
  "manifest config hash matches current config",
  manifestConfigSource?.sha256 === fileSha256(configPath),
  { recorded: manifestConfigSource?.sha256, computed: fileSha256(configPath) },
);

const pass = checks.every((entry) => entry.pass);
process.stdout.write(`${JSON.stringify({ schema_version: 1, workflow: "H7_OFFLINE_RULES_ONLY", pass, checks }, null, 2)}\n`);
process.exit(pass ? 0 : 1);
