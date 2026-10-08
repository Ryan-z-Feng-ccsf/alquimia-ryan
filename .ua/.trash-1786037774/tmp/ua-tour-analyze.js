#!/usr/bin/env node

const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) fail('Usage: ua-tour-analyze.js INPUT OUTPUT');

  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = Array.isArray(input.nodes) ? input.nodes : [];
  const edges = Array.isArray(input.edges) ? input.edges : [];
  const layers = Array.isArray(input.layers) ? input.layers : [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    if (nodeById.has(edge.source) && nodeById.has(edge.target)) {
      fanOut.set(edge.source, fanOut.get(edge.source) + 1);
      fanIn.set(edge.target, fanIn.get(edge.target) + 1);
    }
  }

  const rank = (counts, field) => nodes
    .map((node) => ({id: node.id, [field]: counts.get(node.id), name: node.name}))
    .sort((a, b) => b[field] - a[field] || a.id.localeCompare(b.id))
    .slice(0, 20);
  const fanInRanking = rank(fanIn, 'fanIn');
  const fanOutRanking = rank(fanOut, 'fanOut');

  const codeNodes = nodes.filter((node) => node.type === 'file');
  const fanOutValues = codeNodes.map((node) => fanOut.get(node.id)).sort((a, b) => b - a);
  const topTenThreshold = fanOutValues.length
    ? fanOutValues[Math.max(0, Math.ceil(fanOutValues.length * 0.1) - 1)]
    : Infinity;
  const fanInValues = codeNodes.map((node) => fanIn.get(node.id)).sort((a, b) => a - b);
  const bottomQuarterThreshold = fanInValues.length
    ? fanInValues[Math.max(0, Math.ceil(fanInValues.length * 0.25) - 1)]
    : -1;
  const entryNames = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
    'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
    'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
    'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
    'App.swift', 'Application.kt', 'main.cpp', 'main.c'
  ]);
  const entryPointCandidates = nodes.map((node) => {
    const path = node.filePath || '';
    const name = node.name || path.split('/').pop() || '';
    const depth = path.split('/').filter(Boolean).length;
    let score = 0;
    if (node.type === 'file') {
      if (entryNames.has(name)) score += 3;
      if (depth <= 2) score += 1;
      if (fanOut.get(node.id) >= topTenThreshold) score += 1;
      if (fanIn.get(node.id) <= bottomQuarterThreshold) score += 1;
    } else if (node.type === 'document') {
      if (path === 'README.md') score += 5;
      else if (depth === 1 && path.toLowerCase().endsWith('.md')) score += 2;
    }
    return {id: node.id, score, name, summary: node.summary};
  }).filter((candidate) => candidate.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, 5);

  const topCodeEntry = entryPointCandidates.find((candidate) => nodeById.get(candidate.id)?.type === 'file')
    || codeNodes.map((node) => ({
      id: node.id,
      score: fanOut.get(node.id) - fanIn.get(node.id),
      name: node.name,
      summary: node.summary
    })).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))[0];
  const adjacency = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) {
    if ((edge.type === 'imports' || edge.type === 'calls')
      && nodeById.has(edge.source) && nodeById.has(edge.target)) {
      adjacency.get(edge.source).push(edge.target);
    }
  }
  for (const targets of adjacency.values()) targets.sort();
  const order = [];
  const depthMap = {};
  const byDepth = {};
  if (topCodeEntry) {
    const queue = [topCodeEntry.id];
    depthMap[topCodeEntry.id] = 0;
    while (queue.length) {
      const current = queue.shift();
      order.push(current);
      const depth = depthMap[current];
      if (!byDepth[depth]) byDepth[depth] = [];
      byDepth[depth].push(current);
      for (const target of adjacency.get(current) || []) {
        if (depthMap[target] === undefined) {
          depthMap[target] = depth + 1;
          queue.push(target);
        }
      }
    }
  }

  const item = (node) => ({id: node.id, name: node.name, type: node.type, summary: node.summary});
  const nonCodeFiles = {
    documentation: nodes.filter((node) => node.type === 'document').map(item),
    infrastructure: nodes.filter((node) => ['service', 'pipeline', 'resource'].includes(node.type)).map(item),
    data: nodes.filter((node) => ['table', 'schema', 'endpoint'].includes(node.type)).map(item),
    config: nodes.filter((node) => node.type === 'config').map(item)
  };

  const directed = new Set();
  for (const edge of edges) {
    if ((edge.type === 'imports' || edge.type === 'calls')
      && nodeById.has(edge.source) && nodeById.has(edge.target)) {
      directed.add(`${edge.type}\u0000${edge.source}\u0000${edge.target}`);
    }
  }
  const pairEdges = [];
  for (const edge of edges) {
    if ((edge.type === 'imports' || edge.type === 'calls')
      && nodeById.has(edge.source) && nodeById.has(edge.target)
      && edge.source < edge.target
      && directed.has(`${edge.type}\u0000${edge.target}\u0000${edge.source}`)) {
      pairEdges.push([edge.source, edge.target]);
    }
  }
  const clusters = [];
  const seenPairs = new Set();
  const undirectedNeighbors = new Map(nodes.map((node) => [node.id, new Set()]));
  for (const edge of edges) {
    if (nodeById.has(edge.source) && nodeById.has(edge.target)) {
      undirectedNeighbors.get(edge.source).add(edge.target);
      undirectedNeighbors.get(edge.target).add(edge.source);
    }
  }
  for (const [left, right] of pairEdges) {
    const pairKey = `${left}\u0000${right}`;
    if (seenPairs.has(pairKey)) continue;
    seenPairs.add(pairKey);
    const cluster = [left, right];
    const candidates = nodes
      .map((node) => ({
        id: node.id,
        connections: cluster.filter((member) => undirectedNeighbors.get(member).has(node.id)).length
      }))
      .filter((candidate) => !cluster.includes(candidate.id) && candidate.connections >= 2)
      .sort((a, b) => b.connections - a.connections || a.id.localeCompare(b.id));
    for (const candidate of candidates) {
      if (cluster.length >= 5) break;
      cluster.push(candidate.id);
    }
    const clusterSet = new Set(cluster);
    const edgeCount = edges.filter((edge) => clusterSet.has(edge.source) && clusterSet.has(edge.target)).length;
    clusters.push({nodes: cluster, edgeCount});
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount || a.nodes[0].localeCompare(b.nodes[0]));

  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [node.id, {
    name: node.name,
    type: node.type,
    summary: node.summary
  }]));
  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: {startNode: topCodeEntry?.id || null, order, depthMap, byDepth},
    nonCodeFiles,
    clusters: clusters.slice(0, 10),
    layers: {count: layers.length, list: layers.map(({id, name, description}) => ({id, name, description}))},
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
} catch (error) {
  fail(error.stack || error.message);
}
