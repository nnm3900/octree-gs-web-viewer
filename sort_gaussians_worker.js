function sortSplats(matrices, view) {
  const vertexCount = matrices.length / 16;

  let maxDepth = -Infinity;
  let minDepth = Infinity;
  let depthList = new Float32Array(vertexCount);
  let sizeList = new Int32Array(depthList.buffer);
  for (let i = 0; i < vertexCount; i++) {
    let depth =
      view[0] * matrices[i * 16 + 12] -
      view[1] * matrices[i * 16 + 13] -
      view[2] * matrices[i * 16 + 14];
    depthList[i] = depth;
    if (depth > maxDepth) maxDepth = depth;
    if (depth < minDepth) minDepth = depth;
  }

  let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
  let counts0 = new Uint32Array(256 * 256);
  for (let i = 0; i < vertexCount; i++) {
    sizeList[i] = ((depthList[i] - minDepth) * depthInv) | 0;
    counts0[sizeList[i]]++;
  }
  let starts0 = new Uint32Array(256 * 256);
  for (let i = 1; i < 256 * 256; i++)
    starts0[i] = starts0[i - 1] + counts0[i - 1];
  let depthIndex = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;

  let sortedMatrices = new Float32Array(vertexCount * 16);
  for (let j = 0; j < vertexCount; j++) {
    let i = depthIndex[j];
    for (let k = 0; k < 16; k++) {
      sortedMatrices[j * 16 + k] = matrices[i * 16 + k];
    }
  }
  //console.log(view);
  if(0)if (matrices[0] == sortedMatrices[0]) {
    console.log(0);
  }else{
    console.log(1);
  }
  return sortedMatrices;
}

onmessage = (e) => {
  const { matrices, view } = e.data;
  const sortedMatrices = sortSplats(new Float32Array(matrices), new Float32Array(view));
  postMessage({ sortedMatrices }, [sortedMatrices.buffer]);
};