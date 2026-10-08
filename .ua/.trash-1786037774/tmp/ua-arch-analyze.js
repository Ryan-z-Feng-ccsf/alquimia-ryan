const fs = require('fs');
const path = require('path');

function fail(error) {
  process.stderr.write(`${error.stack || error.message || error}\n`);
  process.exit(1);
}

function commonDirectoryPrefix(paths) {
  const directories = paths.map((filePath) => filePath.split('/').slice(0, -1));
  if (!directories.length) return [];
  const prefix = [];
  for (let index = 0; ; index += 1) {
    const segment = directories[0][index];
    if (!segment || !directories.every((parts) => parts[index] === segment)) break;
    prefix.push(segment);
  }
  return prefix;
}

function flatPattern(filePath, type) {
  const base = path.posix.basename(filePath);
  const lower = base.toLowerCase();
  if (/([._-](test|spec)\.|^test_)/i.test(base)) return 'test';
  if (type === 'config' || /(^|[._-])(config|settings)([._-]|$)/i.test(base)) return 'config';
  if (type === 'document' || /\.(md|rst)$/i.test(base)) return 'documentation';
  const extension = path.posix.extname(lower).replace(/^\./, '');
  return extension || type || 'root';
}

function knownPattern(group, filePaths) {
  const directoryPatterns = new Map([
    ['routes', 'api'], ['api', 'api'], ['controllers', 'api'], ['endpoints', 'api'],
    ['handlers', 'api'], ['serializers', 'api'], ['routers', 'api'], ['blueprints', 'api'],
    ['services', 'service'], ['core', 'service'], ['lib', 'service'], ['domain', 'service'],
    ['logic', 'service'], ['internal', 'service'], ['signals', 'service'], ['composables', 'service'],
    ['mailers', 'service'], ['jobs', 'service'], ['channels', 'service'],
    ['models', 'data'], ['db', 'data'], ['data', 'data'], ['persistence', 'data'],
    ['repository', 'data'], ['entities', 'data'], ['entity', 'data'], ['migrations', 'data'],
    ['database', 'data'], ['sql', 'data'], ['schema', 'data'],
    ['components', 'ui'], ['views', 'ui'], ['pages', 'ui'], ['ui', 'ui'],
    ['layouts', 'ui'], ['screens', 'ui'], ['middleware', 'middleware'], ['plugins', 'middleware'],
    ['interceptors', 'middleware'], ['guards', 'middleware'], ['utils', 'utility'],
    ['helpers', 'utility'], ['common', 'utility'], ['shared', 'utility'], ['tools', 'utility'],
    ['pkg', 'utility'], ['templatetags', 'utility'], ['config', 'config'], ['constants', 'config'],
    ['env', 'config'], ['settings', 'config'], ['management', 'config'], ['commands', 'config'],
    ['__tests__', 'test'], ['test', 'test'], ['tests', 'test'], ['spec', 'test'],
    ['specs', 'test'], ['types', 'types'], ['interfaces', 'types'], ['schemas', 'types'],
    ['contracts', 'types'], ['dtos', 'types'], ['dto', 'types'], ['request', 'types'],
    ['response', 'types'], ['hooks', 'hooks'], ['store', 'state'], ['state', 'state'],
    ['reducers', 'state'], ['actions', 'state'], ['slices', 'state'], ['assets', 'assets'],
    ['static', 'assets'], ['public', 'assets'], ['cmd', 'entry'], ['bin', 'entry'],
    ['docs', 'documentation'], ['documentation', 'documentation'], ['wiki', 'documentation'],
    ['deploy', 'infrastructure'], ['deployment', 'infrastructure'], ['infra', 'infrastructure'],
    ['infrastructure', 'infrastructure'], ['k8s', 'infrastructure'], ['kubernetes', 'infrastructure'],
    ['helm', 'infrastructure'], ['charts', 'infrastructure'], ['terraform', 'infrastructure'],
    ['tf', 'infrastructure'], ['docker', 'infrastructure'], ['.github', 'ci-cd'],
    ['.gitlab', 'ci-cd'], ['.circleci', 'ci-cd'],
  ]);
  const lowerGroup = group.toLowerCase();
  if (directoryPatterns.has(lowerGroup)) return directoryPatterns.get(lowerGroup);
  for (const filePath of filePaths) {
    const base = path.posix.basename(filePath);
    const lower = filePath.toLowerCase();
    if (/([._-](test|spec)\.|^test_|_test\.go$|test\.java$|_spec\.rb$|test\.php$|tests\.cs$)/i.test(base)) return 'test';
    if (/\.d\.ts$/i.test(base) || /\.(graphql|gql|proto)$/i.test(base)) return 'types';
    if (/^(index\.(ts|js)|__init__\.py|manage\.py|config\.ru|application\.java|program\.cs)$/i.test(base)) return 'entry';
    if (/^(wsgi|asgi)\.py$/i.test(base)) return 'config';
    if (/^(cargo\.toml|go\.mod|gemfile|pom\.xml|build\.gradle|composer\.json)$/i.test(base)) return 'config';
    if (/^dockerfile/i.test(base) || /^docker-compose\./i.test(base) || /\.(tf|tfvars)$/i.test(base)) return 'infrastructure';
    if (lower.startsWith('.github/workflows/') || /(^|\/)(\.gitlab-ci\.yml|jenkinsfile)$/i.test(lower)) return 'ci-cd';
    if (/\.sql$/i.test(base)) return 'data';
    if (/\.(md|rst)$/i.test(base)) return 'documentation';
    if (base === 'Makefile') return 'infrastructure';
  }
  return null;
}

try {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) throw new Error('Usage: ua-arch-analyze.js INPUT OUTPUT');
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const fileNodes = input.fileNodes || [];
  const importEdges = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const nodeById = new Map(fileNodes.map((node) => [node.id, node]));
  const paths = fileNodes.map((node) => node.filePath || node.name || node.id);
  const prefix = commonDirectoryPrefix(paths);
  const hasSubdirectories = paths.some((filePath) => filePath.includes('/'));
  const flat = !hasSubdirectories;
  const groupForNode = new Map();
  const directoryGroups = {};
  for (const node of fileNodes) {
    const filePath = node.filePath || node.name || node.id;
    const parts = filePath.split('/');
    let group;
    if (flat) {
      group = flatPattern(filePath, node.type);
    } else if (prefix.length && parts.length > prefix.length + 1) {
      group = parts[prefix.length];
    } else if (parts.length > 1) {
      group = prefix.length && parts.length === prefix.length + 1 ? 'root' : parts[0];
    } else {
      group = 'root';
    }
    groupForNode.set(node.id, group);
    (directoryGroups[group] ||= []).push(node.id);
  }

  const nodeTypeGroups = {};
  for (const node of fileNodes) (nodeTypeGroups[node.type] ||= []).push(node.id);

  const adjacency = Object.fromEntries(fileNodes.map((node) => [node.id, []]));
  const fileFanIn = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const groupImportsFrom = {};
  const groupImportedBy = {};
  const interCounts = new Map();
  const internalCounts = Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, 0]));
  const involvingCounts = Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, 0]));
  for (const edge of importEdges) {
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    adjacency[edge.source].push(edge.target);
    fileFanOut[edge.source] += 1;
    fileFanIn[edge.target] += 1;
    const from = groupForNode.get(edge.source);
    const to = groupForNode.get(edge.target);
    if (from === to) {
      internalCounts[from] += 1;
      involvingCounts[from] += 1;
    } else {
      involvingCounts[from] += 1;
      involvingCounts[to] += 1;
      (groupImportsFrom[from] ||= new Set()).add(to);
      (groupImportedBy[to] ||= new Set()).add(from);
      const key = `${from}\u0000${to}`;
      interCounts.set(key, (interCounts.get(key) || 0) + 1);
    }
  }
  const interGroupImports = [...interCounts.entries()].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return {from, to, count};
  }).sort((a, b) => b.count - a.count || a.from.localeCompare(b.from) || a.to.localeCompare(b.to));
  const groupAdjacency = {};
  for (const group of Object.keys(directoryGroups)) {
    groupAdjacency[group] = {
      importsFrom: [...(groupImportsFrom[group] || [])].sort(),
      importedBy: [...(groupImportedBy[group] || [])].sort(),
    };
  }

  const crossCounts = new Map();
  const nonCodeConnections = {};
  for (const edge of allEdges) {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (!source || !target) continue;
    const key = `${source.type}\u0000${target.type}\u0000${edge.type}`;
    crossCounts.set(key, (crossCounts.get(key) || 0) + 1);
    if (source.type !== 'file' || target.type !== 'file') {
      (nonCodeConnections[edge.source] ||= []).push({target: edge.target, edgeType: edge.type});
    }
  }
  const crossCategoryEdges = [...crossCounts.entries()].map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('\u0000');
    return {fromType, toType, edgeType, count};
  }).sort((a, b) => b.count - a.count);

  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    const internalEdges = internalCounts[group];
    const totalEdges = involvingCounts[group];
    intraGroupDensity[group] = {
      internalEdges,
      totalEdges,
      density: totalEdges ? internalEdges / totalEdges : 0,
    };
  }
  const patternMatches = {};
  for (const [group, ids] of Object.entries(directoryGroups)) {
    const match = knownPattern(group, ids.map((id) => nodeById.get(id).filePath));
    if (match) patternMatches[group] = match;
  }

  const lowerPaths = paths.map((filePath) => filePath.toLowerCase());
  const infraFiles = paths.filter((filePath) => {
    const lower = filePath.toLowerCase();
    const base = path.posix.basename(lower);
    return /^dockerfile/.test(base) || /^docker-compose\./.test(base) || /\.(tf|tfvars)$/.test(base) ||
      lower.includes('/k8s/') || lower.includes('/kubernetes/') || lower.includes('/helm/') ||
      lower.startsWith('.github/workflows/') || base === '.gitlab-ci.yml' || base === 'jenkinsfile' || base === 'makefile';
  });
  const deploymentTopology = {
    hasDockerfile: lowerPaths.some((value) => /^dockerfile/.test(path.posix.basename(value))),
    hasCompose: lowerPaths.some((value) => /^docker-compose\./.test(path.posix.basename(value))),
    hasK8s: lowerPaths.some((value) => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/.test(value)),
    hasTerraform: lowerPaths.some((value) => /\.(tf|tfvars)$/.test(value)),
    hasCI: lowerPaths.some((value) => value.startsWith('.github/workflows/') || /(^|\/)(\.gitlab-ci\.yml|jenkinsfile)$/.test(value)),
    infraFiles,
  };
  const dataPipeline = {
    schemaFiles: paths.filter((value) => /\.(sql|graphql|gql|proto|prisma)$/i.test(value)),
    migrationFiles: paths.filter((value) => /(^|\/)migrations?\//i.test(value)),
    dataModelFiles: paths.filter((value) => /(^|\/)(models?|entities|repository|repositories)\//i.test(value)),
    apiHandlerFiles: paths.filter((value) => /(^|\/)(routes?|controllers?|handlers?|endpoints?)\//i.test(value)),
  };

  const docNodes = fileNodes.filter((node) => node.type === 'document' || /\.(md|rst)$/i.test(node.filePath || ''));
  const groupsWithDocs = new Set();
  for (const node of docNodes) {
    const ownGroup = groupForNode.get(node.id);
    if (ownGroup) groupsWithDocs.add(ownGroup);
    const text = `${node.summary || ''} ${(node.tags || []).join(' ')}`.toLowerCase();
    for (const group of Object.keys(directoryGroups)) {
      if (text.includes(group.toLowerCase())) groupsWithDocs.add(group);
    }
  }
  const allGroups = Object.keys(directoryGroups);
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.size,
    totalGroups: allGroups.length,
    coverageRatio: allGroups.length ? groupsWithDocs.size / allGroups.length : 0,
    undocumentedGroups: allGroups.filter((group) => !groupsWithDocs.has(group)),
  };

  const pairCounts = new Map();
  for (const item of interGroupImports) {
    const pair = [item.from, item.to].sort();
    const key = pair.join('\u0000');
    const counts = pairCounts.get(key) || {};
    counts[`${item.from}\u0000${item.to}`] = item.count;
    pairCounts.set(key, counts);
  }
  const dependencyDirection = [];
  for (const [key, counts] of pairCounts) {
    const [a, b] = key.split('\u0000');
    const ab = counts[`${a}\u0000${b}`] || 0;
    const ba = counts[`${b}\u0000${a}`] || 0;
    if (ab > ba) dependencyDirection.push({dependent: a, dependsOn: b});
    else if (ba > ab) dependencyDirection.push({dependent: b, dependsOn: a});
    else if (ab) dependencyDirection.push({dependent: a, dependsOn: b, bidirectional: true});
  }

  const output = {
    scriptCompleted: true,
    commonPathPrefix: prefix.join('/'),
    directoryGroups,
    nodeTypeGroups,
    importAdjacency: adjacency,
    groupAdjacency,
    crossCategoryEdges,
    nonCodeConnections,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([group, ids]) => [group, ids.length])),
      nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([type, ids]) => [type, ids.length])),
    },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`);
} catch (error) {
  fail(error);
}
