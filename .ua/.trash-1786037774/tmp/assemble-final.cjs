#!/usr/bin/env node
const fs = require('fs');

const projectRoot = process.argv[2];
const uaDir = `${projectRoot}/.ua`;
const scan = JSON.parse(fs.readFileSync(`${uaDir}/intermediate/scan-result.json`, 'utf8'));
const assembled = JSON.parse(fs.readFileSync(`${uaDir}/intermediate/assembled-graph.json`, 'utf8'));
const layers = JSON.parse(fs.readFileSync(`${uaDir}/intermediate/layers.json`, 'utf8'));
const tour = JSON.parse(fs.readFileSync(`${uaDir}/intermediate/tour.json`, 'utf8'));
const gitCommitHash = process.argv[3];

const graph = {
  version: '1.0.0',
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt: new Date().toISOString(),
    gitCommitHash,
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers,
  tour,
};

fs.writeFileSync(
  `${uaDir}/intermediate/assembled-graph.json`,
  `${JSON.stringify(graph, null, 2)}\n`,
);
