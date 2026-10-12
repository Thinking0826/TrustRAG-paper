// Synthetic controller examples, not real human labels or browser rendering.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");
const code = fs.readFileSync("src/trustrag/qasper_review.html", "utf8")
  .match(/<script>\s*([\s\S]*?)<\/script>/)[1];

class Element {
  constructor() { this.children = []; this.value = ""; this.textContent = ""; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  click() { if (this.onclick) this.onclick(); }
}

function page({storageFailure = false} = {}) {
  const packet = {
    study_id: "synthetic", schema_version: "qasper_blind_review_v1", review_count: 2,
    cases: ["answered", "rejected"].map((outcome, i) => ({
      case_id: "R00" + (i + 1), document_id: "doc", question: "Question " + i,
      outcome, displayed_answer: outcome === "answered" ? "Answer [E1]" : null,
      raw_candidate: "Raw candidate [E1]", alternative_references: [],
      evidence: [{citation_id: "E1", selected_text: "short", original_paragraph_ids: ["p1"]}]
    })),
    documents: {doc: {title: "Paper", abstract: "Abstract", paragraphs: [
      {paragraph_id: "p1", section: "Methods", text: "Full original paragraph"}
    ]}}
  };
  const elements = new Map(), downloads = [], blobs = new Map(), alerts = [];
  const document = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement(tag) {
      const e = new Element();
      if (tag === "a") e.onclick = () => downloads.push({url: e.href, filename: e.download});
      return e;
    }
  };
  document.getElementById("packet").textContent = JSON.stringify(packet);
  const stored = new Map();
  const context = vm.createContext({document, Blob,
    localStorage: {
      getItem(key) { if (storageFailure) throw Error("unavailable"); return stored.get(key); },
      setItem(key, value) { if (storageFailure) throw Error("unavailable"); stored.set(key, value); }
    },
    URL: {createObjectURL(blob) { const key = "blob:" + blobs.size; blobs.set(key, blob); return key; }, revokeObjectURL() {}},
    alert(message) { alerts.push(message); }, confirm() { return true; }, setTimeout(callback) { callback(); }
  });
  vm.runInContext(code, context);
  return {elements, downloads, blobs, alerts,
    evaluate: value => vm.runInContext(value, context),
    get: id => document.getElementById(id)};
}

test("initial blank judgments, source resolution and rejected answer separation", () => {
  const p = page();
  assert.equal(p.get("progress").textContent, "0 / 2 已填写");
  assert.match(p.get("evidence").children[0].children.at(-1).textContent, /Full original paragraph/);
  p.get("next").click();
  assert.equal(p.get("answer").textContent, "未展示有效回答");
  assert.equal(p.get("raw").textContent, "Raw candidate [E1]");
  assert.match(p.get("outcome").textContent, /拒收/);
});

test("judgments and notes survive navigation; reviewer ID may be entered before export", () => {
  const p = page();
  const select = p.get("ratings").children[0].children[0];
  select.value = "unsure"; select.onchange();
  p.get("notes").value = "Synthetic test only"; p.get("notes").oninput();
  p.get("reviewer").value = "fixture_A"; p.get("reviewer").onchange();
  p.get("next").click(); p.get("previous").click();
  p.get("jump").value = p.evaluate('reviewCases.findIndex(c=>c.case_id==="R001")');
  p.get("jump").onchange();
  assert.equal(p.get("ratings").children[0].children[0].value, "unsure");
  assert.equal(p.get("notes").value, "Synthetic test only");
  assert.equal(p.evaluate("state.reviewer_id"), "fixture_A");
});

test("export retains incomplete and uncertainty instead of manufacturing correctness", async () => {
  const p = page();
  p.get("export").click();
  assert.equal(p.downloads.length, 0);
  p.get("reviewer").value = "fixture_A"; p.get("reviewer").onchange();
  p.evaluate('state.ratings.R001={relevant:"yes",factual:"unsure",complete:"no",citations_supported:"no",unsupported_additions:"yes"}');
  p.get("export").click();
  const value = JSON.parse(await p.blobs.get(p.downloads[0].url).text());
  assert.equal(value.reviewed_count, 1);
  assert.equal(value.complete, false);
  assert.equal(value.independent, false);
  assert.equal(value.ratings.R001.factual, "unsure");
});

test("disabled browser storage still permits actual record download", () => {
  const p = page({storageFailure: true});
  p.get("reviewer").value = "fixture_A"; p.get("reviewer").onchange();
  assert.match(p.get("save").textContent, /未允许缓存/);
  p.get("export").click();
  assert.equal(p.downloads.length, 1);
});

test("reviewer ordering preserves IDs and is stable across sessions", () => {
  const p = page();
  assert.equal(p.evaluate('JSON.stringify(orderForReviewer("reviewer_01"))'),
    p.evaluate('JSON.stringify(orderForReviewer("reviewer_01"))'));
  assert.equal(p.evaluate('orderForReviewer("reviewer_05").map(c=>c.case_id).sort().join(",")'), "R001,R002");
  p.get("reviewer").value = "reviewer_01"; p.get("reviewer").onchange();
  assert.equal(p.evaluate("Object.keys(state.ratings).length"), 0);
  assert.equal(p.evaluate("state.rubric_version"), "five_reviewers_v2");
});

test("negative polarity and uncertainty notes are explained without prefilling ratings", () => {
  const p = page();
  assert.match(p.get("ratings").children[4].textContent, /负向项/);
  const select = p.get("ratings").children[1].children[0];
  select.value = "unsure"; select.onchange();
  assert.match(p.get("rating-hint").textContent, /具体原因/);
  p.get("notes").value = "P1 and P2 disagree"; p.get("notes").oninput();
  assert.equal(p.get("rating-hint").textContent, "");
});
