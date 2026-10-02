// Exercise the actual chat renderer without a browser or external packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse the entire dashboard script too.
const renderer = script.slice(script.indexOf('function renderChat'), script.indexOf('function renderActions'));

class Element {
  constructor() {
    this.children = [];
    this.attributes = {};
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.clientHeight = 0;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
}

const messages = new Element();
const context = vm.createContext({messages, document: {createElement: () => new Element()}});
vm.runInContext("let chatVersion='';" + renderer, context);

for (const [status, label] of Object.entries({
  queued: 'Queued', thinking: 'Thinking', planned: 'Plan ready',
  executing: 'Executing', completed: 'Completed', failed: 'Failed',
  cancelled: 'Cancelled', error: 'Error'
})) {
  const items = [
    {id: 1, role: 'user', content: 'Go to the red chair', status},
    {id: 1, role: 'assistant', content: 'Search for and approach red chair.', status}
  ];
  vm.runInContext(`renderChat(${JSON.stringify(items)})`, context);
  assert.equal(messages.children.length, 2);
  for (const [index, entry] of messages.children.entries()) {
    const [bubble, meta] = entry.children;
    const [role, badge] = meta.children;
    assert.equal(bubble.textContent, items[index].content);
    assert.equal(role.textContent, index === 0 ? 'You' : 'Robot');
    assert.equal(badge.textContent, label);
    assert.equal(badge.className, 'task-status ' + status);
    assert.equal(badge.attributes['aria-label'], 'Task status: ' + label);
  }
  const before = messages.children[0];
  vm.runInContext(`renderChat(${JSON.stringify(items)})`, context);
  assert.equal(messages.children[0], before, 'Unchanged chat should retain its DOM');
}
console.log('Chat renderer status badge tests passed');

// Exercise the actual state poll so all model labels reflect the server values.
const pollSource = script.slice(script.indexOf('async function poll'), script.indexOf('async function post'));
const modelElements = new Map(['llm-model', 'vlm-model', 'vision-model', 'connection'].map(id => [id, new Element()]));
let dashboardState = {
  model: 'planner-model', vlm_model: 'gpt-4o', vision_model: 'yolo11n.pt',
  messages: [], actions: [], active_action: null, logs: [], running: true
};
const pollContext = vm.createContext({
  document: {getElementById: id => modelElements.get(id)},
  fetch: async () => ({ok: true, json: async () => dashboardState}),
  renderChat: () => {}, renderActions: () => {}
});
vm.runInContext('let polling=false,ended=false,logId=0;' + pollSource, pollContext);
(async () => {
  await vm.runInContext('poll()', pollContext);
  assert.equal(modelElements.get('llm-model').textContent, 'planner-model');
  assert.equal(modelElements.get('vlm-model').textContent, 'gpt-4o');
  assert.equal(modelElements.get('vision-model').textContent, 'yolo11n.pt');
  dashboardState = {...dashboardState, vlm_model: 'gpt-6-luna', vision_model: '/tmp/custom weights.pt'};
  await vm.runInContext('poll()', pollContext);
  assert.equal(modelElements.get('vlm-model').textContent, 'gpt-6-luna');
  assert.equal(modelElements.get('vision-model').textContent, '/tmp/custom weights.pt');
  console.log('Dashboard model label tests passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
