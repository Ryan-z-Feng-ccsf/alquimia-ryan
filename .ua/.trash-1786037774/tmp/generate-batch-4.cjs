const fs = require('fs');
const path = require('path');

const root = '/home/zfeng3/alquimia-ryan';
const ua = path.join(root, '.ua');
const extraction = JSON.parse(fs.readFileSync(path.join(ua, 'tmp/ua-file-extract-results-4.json'), 'utf8'));
const batches = JSON.parse(fs.readFileSync(path.join(ua, 'intermediate/batches.json'), 'utf8'));
const batch = batches.batches.find((item) => item.batchIndex === 4);

const fileDetails = {
  'alquimia/alquimia_constants.c': {
    summary: 'Defines shared string limits, geochemical quantity names, engine identifiers, and error codes used throughout the Alquimia C interface.',
    tags: ['constants', 'geochemistry', 'error-codes', 'shared-api'],
  },
  'alquimia/alquimia_constants.h': {
    summary: 'Declares the public constants for supported engines, geochemical quantities, string limits, and Alquimia error codes.',
    tags: ['type-definition', 'constants', 'public-api', 'geochemistry'],
  },
  'alquimia/alquimia_containers.h': {
    summary: 'Defines the C-compatible vectors and geochemical state, property, metadata, condition, constraint, and engine-status structures shared with reaction engines and Fortran bindings.',
    tags: ['data-model', 'type-definition', 'geochemistry', 'interop'],
    languageNotes: 'The plain C structs and extern C guards preserve a stable layout and calling boundary for C, C++, and Fortran interoperability.',
  },
  'alquimia/alquimia_interface.h': {
    summary: 'Defines the engine-neutral AlquimiaData aggregate and the AlquimiaInterface function-pointer contract for setup, reaction stepping, condition processing, metadata, and auxiliary output.',
    tags: ['public-api', 'interface', 'function-pointers', 'engine-adapter'],
    languageNotes: 'A C function-pointer table provides runtime engine polymorphism without exposing engine-specific types to drivers.',
  },
  'alquimia/alquimia_memory.c': {
    summary: 'Implements allocation and release helpers for every dynamically sized Alquimia vector, state, metadata, condition, constraint, and aggregate data structure.',
    tags: ['memory-management', 'resource-lifecycle', 'data-model', 'utility'],
    languageNotes: 'Vector capacities are rounded to powers of two, and paired Allocate/Free routines centralize ownership of nested C allocations.',
  },
  'alquimia/alquimia_util.c': {
    summary: 'Provides shared comparison, lookup, deep-copy, resize, and diagnostic printing operations for Alquimia containers and geochemical data structures.',
    tags: ['utility', 'serialization', 'data-model', 'diagnostics'],
    languageNotes: 'The utility API applies operation families consistently across primitive vectors and nested geochemical container types.',
  },
};

function complexity(nonEmptyLines) {
  if (nonEmptyLines > 200) return 'complex';
  if (nonEmptyLines >= 50) return 'moderate';
  return 'simple';
}

function objectLabel(name) {
  return name
    .replace(/^(Allocate|Free|Copy|Resize|Print)/, '')
    .replace(/^Alquimia/, 'Alquimia ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/Vector Double/, 'double vector')
    .replace(/Vector Int/, 'integer vector')
    .replace(/Vector String/, 'string vector')
    .replace(/Auxiliary Output Data/, 'auxiliary output data')
    .replace(/Auxiliary Data/, 'auxiliary data')
    .replace(/Problem Meta Data/, 'problem metadata')
    .replace(/Engine Functionality/, 'engine functionality')
    .replace(/Engine Status/, 'engine status')
    .replace(/Geochemical Condition Vector/, 'geochemical-condition vector')
    .replace(/Geochemical Condition/, 'geochemical condition')
    .replace(/Aqueous Constraint Vector/, 'aqueous-constraint vector')
    .replace(/Aqueous Constraint/, 'aqueous constraint')
    .replace(/Mineral Constraint Vector/, 'mineral-constraint vector')
    .replace(/Mineral Constraint/, 'mineral constraint')
    .replace(/Properties/, 'properties')
    .replace(/State/, 'state')
    .replace(/Sizes/, 'sizes')
    .replace(/Data/, 'data aggregate')
    .trim()
    .replace(/^Alquimia /, 'Alquimia ');
}

function functionSummary(name) {
  const label = objectLabel(name);
  if (name === 'AlquimiaCaseInsensitiveStringCompare') {
    return 'Compares two strings without case sensitivity and returns the ordering result used by name-based lookup operations.';
  }
  if (name === 'AlquimiaFindIndexFromName') {
    return 'Finds a requested name in an Alquimia string vector using case-insensitive comparison and returns its index or a missing-value sentinel.';
  }
  if (name === 'AllocateAlquimiaVectorDouble' || name === 'AllocateAlquimiaVectorInt') {
    return `Allocates a zero-initialized ${label} with power-of-two capacity while retaining the requested logical size.`;
  }
  if (name === 'AllocateAlquimiaVectorString') {
    return 'Allocates a string vector and fixed-size character buffers for every requested element.';
  }
  if (name.startsWith('Allocate')) {
    return `Allocates and initializes the nested storage required by an ${label} according to the supplied sizes or element count.`;
  }
  if (name.startsWith('Free')) {
    return `Releases owned storage held by an ${label} and clears its dynamically allocated members.`;
  }
  if (name.startsWith('Copy')) {
    return `Copies the values and nested members of an ${label} into an already allocated destination.`;
  }
  if (name.startsWith('Resize')) {
    return `Resizes an ${label}, preserving existing elements and allocating replacement storage when capacity is insufficient.`;
  }
  if (name.startsWith('Print')) {
    return `Prints the fields and nested values of an ${label} in a human-readable diagnostic form.`;
  }
  return `Implements the ${name} operation for Alquimia container data.`;
}

function functionTags(name) {
  if (name === 'AlquimiaCaseInsensitiveStringCompare') return ['utility', 'string-comparison', 'lookup'];
  if (name === 'AlquimiaFindIndexFromName') return ['utility', 'name-lookup', 'validation'];
  if (name.startsWith('Allocate')) return ['memory-management', 'allocation', 'resource-lifecycle'];
  if (name.startsWith('Free')) return ['memory-management', 'deallocation', 'resource-lifecycle'];
  if (name.startsWith('Copy')) return ['utility', 'deep-copy', 'data-model'];
  if (name.startsWith('Resize')) return ['memory-management', 'resize', 'data-preservation'];
  if (name.startsWith('Print')) return ['diagnostics', 'serialization', 'debugging'];
  return ['utility', 'data-model', 'c-api'];
}

const nodes = [];
const edges = [];
const functionNodes = new Map();

for (const result of extraction.results) {
  const info = fileDetails[result.path];
  nodes.push({
    id: `file:${result.path}`,
    type: 'file',
    name: path.basename(result.path),
    filePath: result.path,
    summary: info.summary,
    tags: info.tags,
    complexity: complexity(result.nonEmptyLines),
    ...(info.languageNotes ? {languageNotes: info.languageNotes} : {}),
  });

  const exported = new Set((result.exports || []).map((item) => item.name));
  for (const fn of result.functions || []) {
    const lines = fn.endLine - fn.startLine + 1;
    if (lines < 10 && !exported.has(fn.name)) continue;
    const id = `function:${result.path}:${fn.name}`;
    functionNodes.set(`${result.path}:${fn.name}`, id);
    nodes.push({
      id,
      type: 'function',
      name: fn.name,
      filePath: result.path,
      lineRange: [fn.startLine, fn.endLine],
      summary: functionSummary(fn.name),
      tags: functionTags(fn.name),
      complexity: lines > 40 ? 'moderate' : 'simple',
    });
    edges.push({source: `file:${result.path}`, target: id, type: 'contains', direction: 'forward', weight: 1.0});
    if (exported.has(fn.name)) {
      edges.push({source: `file:${result.path}`, target: id, type: 'exports', direction: 'forward', weight: 0.8});
    }
  }
}

for (const file of batch.files) {
  for (const target of batch.batchImportData[file.path]) {
    edges.push({source: `file:${file.path}`, target: `file:${target}`, type: 'imports', direction: 'forward', weight: 0.7});
  }
}

const memoryResult = extraction.results.find((item) => item.path === 'alquimia/alquimia_memory.c');
const utilResult = extraction.results.find((item) => item.path === 'alquimia/alquimia_util.c');
const memoryExports = new Set(memoryResult.exports.map((item) => item.name));
const seenCalls = new Set();
for (const call of utilResult.callGraph || []) {
  if (!memoryExports.has(call.callee)) continue;
  const source = functionNodes.get(`alquimia/alquimia_util.c:${call.caller}`);
  const target = functionNodes.get(`alquimia/alquimia_memory.c:${call.callee}`);
  if (!source || !target) continue;
  const key = `${source}|${target}`;
  if (seenCalls.has(key)) continue;
  seenCalls.add(key);
  edges.push({source, target, type: 'calls', direction: 'forward', weight: 0.8});
}

const fileGroups = [
  new Set(batch.files.slice(0, 3).map((item) => item.path)),
  new Set(batch.files.slice(3, 6).map((item) => item.path)),
];

for (let index = 0; index < fileGroups.length; index++) {
  const group = fileGroups[index];
  const partNodes = nodes.filter((node) => group.has(node.filePath));
  const sourceIds = new Set(partNodes.map((node) => node.id));
  const partEdges = edges.filter((edge) => sourceIds.has(edge.source));
  fs.writeFileSync(
    path.join(ua, `intermediate/batch-4-part-${index + 1}.json`),
    JSON.stringify({nodes: partNodes, edges: partEdges}, null, 2) + '\n',
  );
}

console.log(JSON.stringify({nodes: nodes.length, edges: edges.length, imports: edges.filter((edge) => edge.type === 'imports').length}, null, 2));
