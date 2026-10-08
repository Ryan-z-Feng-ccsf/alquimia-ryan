#!/usr/bin/env node
const fs = require('fs');

const projectRoot = process.argv[2];
const outputPath = process.argv[3];
const gitCommitHash = process.argv[4];
const scan = JSON.parse(
  fs.readFileSync(`${projectRoot}/.ua/intermediate/scan-result.json`, 'utf8'),
);

const input = {
  projectRoot,
  sourceFilePaths: scan.files.map(file => file.path),
  gitCommitHash,
};

fs.writeFileSync(outputPath, `${JSON.stringify(input, null, 2)}\n`);
