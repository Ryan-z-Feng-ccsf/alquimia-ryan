const fs = require('fs');

const sourcePath = process.argv[2];
const outputPath = process.argv[3];
const graph = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
const fileLevelTypes = new Set([
  'file', 'config', 'document', 'service', 'pipeline',
  'table', 'schema', 'resource', 'endpoint',
]);
const fileNodes = graph.nodes
  .filter((node) => fileLevelTypes.has(node.type))
  .map(({id, type, name, filePath, summary, tags}) => ({
    id, type, name, filePath, summary, tags,
  }));
const fileNodeIds = new Set(fileNodes.map((node) => node.id));
const fileEdges = graph.edges.filter(
  (edge) => fileNodeIds.has(edge.source) && fileNodeIds.has(edge.target),
);
const input = {
  fileNodes,
  importEdges: fileEdges.filter((edge) => edge.type === 'imports'),
  allEdges: fileEdges,
};
fs.writeFileSync(outputPath, `${JSON.stringify(input, null, 2)}\n`);
