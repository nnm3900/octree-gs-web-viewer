importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let anchor_pos, anchor_scale, anchor_features, anchor_levels, anchor_extra_levels, anchor_info, gsplat_positions;
let anchorCount, vertexCount;
let color_mlp_session, opacity_mlp_session, cov_mlp_session;
let mtx_all;
const gaussian_num_per_anchor = 10;

async function loadFile(url) {
  const response = await fetch(url);
  const data = await response.blob();
  const buffer = await data.arrayBuffer();
  return buffer;
}

async function init() {
  const buffer_anchor_pos = await loadFile("./converted/anchor_positions.bin");
  anchor_pos = new Float32Array(buffer_anchor_pos);
  const buffer_anchor_scale = await loadFile("./converted/anchor_scales.bin");
  anchor_scale = new Float32Array(buffer_anchor_scale);
  const buffer_anchor_features = await loadFile("./converted/anchor_features.bin");
  anchor_features = new Float32Array(buffer_anchor_features);
  const buffer_anchor_levels = await loadFile("./converted/anchor_levels.bin");
  anchor_levels = new Float32Array(buffer_anchor_levels);
  const buffer_anchor_extra_levels = await loadFile("./converted/anchor_extra_levels.bin");
  anchor_extra_levels = new Float32Array(buffer_anchor_extra_levels);
  const buffer_anchor_info = await loadFile("./converted/anchor_info.bin");
  anchor_info = new Float32Array(buffer_anchor_info);
  const buffer_gsplat_positions = await loadFile("./converted/gsplat_positions.bin");
  gsplat_positions = new Float32Array(buffer_gsplat_positions);

  anchorCount = anchor_features.length / 32;
  vertexCount = anchorCount * gaussian_num_per_anchor ;

  const option = { executionProviders: ['webgl'] };
  color_mlp_session = await ort.InferenceSession.create('./converted/color_mlp.ort', option);
  opacity_mlp_session = await ort.InferenceSession.create('./converted/opacity_mlp.ort', option);
  cov_mlp_session = await ort.InferenceSession.create('./converted/cov_mlp.ort', option);

  mtx_all = new Float32Array(anchorCount * gaussian_num_per_anchor * 16).fill(0);

  return vertexCount
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function normalize(x, p = 2, eps = 1e-12) {
  const norm = Math.pow(
    x.reduce((a, b) => a + Math.pow(Math.abs(b), p), 0),
    1 / p
  );
  return x.map(val => val / (norm + eps));
}

function makeQuaternionFromVec(vec) {
  return [vec[1], vec[2], - vec[3], vec[0]];
}
  
function transposeMatrix4(mtx) {
  return [
    mtx[0], mtx[4], mtx[8], mtx[12],
    mtx[1], mtx[5], mtx[9], mtx[13],
    mtx[2], mtx[6], mtx[10], mtx[14],
    mtx[3], mtx[7], mtx[11], mtx[15]
  ];
}

function multiplyMatrix4(a, b) {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = b[0], b01 = b[1], b02 = b[2], b03 = b[3];
  const b10 = b[4], b11 = b[5], b12 = b[6], b13 = b[7];
  const b20 = b[8], b21 = b[9], b22 = b[10], b23 = b[11];
  const b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];
  
  return [
    b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30,
    b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31,
    b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32,
    b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33,
    b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30,
    b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31,
    b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32,
    b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33,
    b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30,
    b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31,
    b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32,
    b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33,
    b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30,
    b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31,
    b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32,
    b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33
  ];
}

function makeMatrix4(position, quaternion, scale, color, opacity) {
  const [x, y, z] = position;
  const [qx, qy, qz, qw] = quaternion;
  const [sx, sy, sz] = scale;  
  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;
  const mtx = [
    (1 - 2 * yy - 2 * zz) * sx, (2 * xy + 2 * wz) * sy, (2 * xz - 2 * wy) * sz, 0,
    (2 * xy - 2 * wz) * sx, (1 - 2 * xx - 2 * zz) * sy, (2 * yz + 2 * wx) * sz, 0,
    (2 * xz + 2 * wy) * sx, (2 * yz - 2 * wx) * sy, (1 - 2 * xx - 2 * yy) * sz, 0,
    0, 0, 0, 1
  ];
  const mtx_t = transposeMatrix4(mtx) 
  const mtx_multiplied = multiplyMatrix4(mtx_t, mtx);
  mtx_multiplied[12] = x;
  mtx_multiplied[13] = y;
  mtx_multiplied[14] = z;
  mtx_multiplied[3] = color[0];
  mtx_multiplied[7] = color[1];
  mtx_multiplied[11] = color[2];
  mtx_multiplied[15] = opacity;
  return mtx_multiplied;
}
  
function multiplyMatrixVector(matrix, vector) {
  const result = [];
  for (let i = 0; i < 4; i++) {
    let sum = 0;
    for (let j = 0; j < 4; j++) {
      sum += matrix[i * 4 + j] * vector[j];
    }
    result[i] = sum;
  }
  return result;  
}
  
function transpose4x4(matrix) {
  return [
    matrix[0], matrix[4], matrix[8], matrix[12],
    matrix[1], matrix[5], matrix[9], matrix[13],
    matrix[2], matrix[6], matrix[10], matrix[14],
    matrix[3], matrix[7], matrix[11], matrix[15]
  ];
}

async function run_onnx(anchor_feature_with_camera, anchorMaskedCount) {
  batchSize = 1000;
  const color_mlp_output = new Float32Array(anchorMaskedCount * 30);
  const opacity_mlp_output = new Float32Array(anchorMaskedCount * 10);
  const cov_mlp_output = new Float32Array(anchorMaskedCount * 70);
  for (let i = 0; i < anchorMaskedCount / batchSize ; i++) {
    let batchData = new Float32Array(35 * batchSize);
    if((i + 1) * 35 * batchSize <= anchorMaskedCount * 35){
      batchData.set(anchor_feature_with_camera.slice(i * 35 * batchSize, (i + 1) * 35 * batchSize), 0);
    }else{
      batchData.set(anchor_feature_with_camera.slice(i * 35 * batchSize, anchorMaskedCount * 35), 0);
      const dummyData = new Array((i + 1) * 35 * batchSize - anchorMaskedCount * 35).fill(0);
      batchData.set(dummyData, anchorMaskedCount * 35 - i * 35 * batchSize);
    }

    const inputTensor = new ort.Tensor('float32', batchData, [batchSize, 1, 1, 35]);
    const color_mlp_outputMap = await color_mlp_session.run({ 'input': inputTensor });
    const opacity_mlp_outputMap = await opacity_mlp_session.run({ 'input': inputTensor });
    const cov_mlp_outputMap = await cov_mlp_session.run({ 'input': inputTensor });
    const color_array =  Array.from(color_mlp_outputMap.output.data)
    const opacity_array =  Array.from(opacity_mlp_outputMap.output.data)
    const cov_array =  Array.from(cov_mlp_outputMap.output.data)
    if(i < Math.floor(anchorMaskedCount / batchSize)){
      color_mlp_output.set(color_array, i * batchSize * 30);
      opacity_mlp_output.set(opacity_array, i * batchSize * 10);
      cov_mlp_output.set(cov_array, i * batchSize * 70);
    }else{
      color_mlp_output.set(color_array.slice(0, (anchorMaskedCount - i * batchSize) * 30), i * batchSize * 30);
      opacity_mlp_output.set(opacity_array.slice(0, (anchorMaskedCount - i * batchSize) * 10), i * batchSize * 10);
      cov_mlp_output.set(cov_array.slice(0, (anchorMaskedCount - i * batchSize) * 70), i * batchSize * 70);
    }
  }
  return [color_mlp_output, opacity_mlp_output, cov_mlp_output];
}

async function computeMtx(camera_position, camera_projectionMatrix, camera_viewMatrix, window_size) {
  const anchorVisibleIndices = [];
  const camera_viewMatrix_t = transpose4x4(camera_viewMatrix);
  const camera_projectionMatrix_t = transpose4x4(camera_projectionMatrix);
  for (let i = 0; i < anchorCount; i++) {
    const x = anchor_pos[i * 3 + 0];
    const y = anchor_pos[i * 3 + 1];
    const z = - anchor_pos[i * 3 + 2];
    const adjm4 = [1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1]
    const vec_0 = [x, y, z, 1];
    const vec_1 = multiplyMatrixVector(camera_viewMatrix_t, vec_0);
    const vec_2 = multiplyMatrixVector(adjm4, vec_1);
    const vec_3 = multiplyMatrixVector(camera_projectionMatrix_t, vec_2);
    const pos2d = vec_3;
    const bounds = 1.2 * pos2d[3];
    if (pos2d[2] < -pos2d[3] || pos2d[0] < -bounds || pos2d[0] > bounds || pos2d[1] < -bounds || pos2d[1] > bounds) {
      continue;
    }else{
      anchorVisibleIndices.push(i);
    }  
  }
  //console.log("visible filter rate", anchorVisibleIndices.length / anchorCount);

  const anchorLevelmaskedIndices = [];
  for (let i = 0; i < anchorVisibleIndices.length; i++) {
    const idx = anchorVisibleIndices[i];
    const dx = anchor_pos[idx * 3 + 0] - camera_position.x;
    const dy = anchor_pos[idx * 3 + 1] - camera_position.y;
    const dz = anchor_pos[idx * 3 + 2] + camera_position.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const pred_level = Math.log2(anchor_info[1]/distance) + anchor_extra_levels[idx]; 
    const int_level = Math.round(pred_level);
    if (anchor_levels[i] <= int_level) {
      anchorLevelmaskedIndices.push(idx);
    }
  }
  //console.log("level filter rate", anchorLevelmaskedIndices.length / anchorVisibleIndices.length, anchorLevelmaskedIndices.length);

  const anchorMaskedIndices =  anchorLevelmaskedIndices;
  const anchorMaskedCount = anchorMaskedIndices.length;

  let anchor_feature_with_camera = new Float32Array(35 * anchorMaskedCount);
  for (let i = 0; i < anchorMaskedCount; i++) {
    const idx = anchorMaskedIndices[i];
    const anchor_feature = anchor_features.slice(idx * 32, (idx + 1) * 32);
    anchor_feature_with_camera.set(anchor_feature, i * 35);
    const dx = anchor_pos[idx * 3 + 0] - camera_position.x;
    const dy = anchor_pos[idx * 3 + 1] - camera_position.y;
    const dz = anchor_pos[idx * 3 + 2] + camera_position.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    anchor_feature_with_camera[i * 35 + 32] = dx / distance;
    anchor_feature_with_camera[i * 35 + 33] = dy / distance;
    anchor_feature_with_camera[i * 35 + 34] = dz / distance;
  }

  const [color_mlp_output, opacity_mlp_output, cov_mlp_output] = await run_onnx(anchor_feature_with_camera, anchorMaskedCount); 

  for (let i = 0; i < anchorMaskedCount * gaussian_num_per_anchor ; i++) {
    const anchor_idx = anchorMaskedIndices[Math.floor(i / 10)];
    const idx = anchor_idx * gaussian_num_per_anchor + i % 10;
    const center = [
      gsplat_positions[3 * idx + 0],
      gsplat_positions[3 * idx + 1],
      -gsplat_positions[3 * idx + 2]
    ];
    const scale = [
      sigmoid(cov_mlp_output[i * 7 + 0]) * anchor_scale[anchor_idx  * 3 + 0],
      sigmoid(cov_mlp_output[i * 7 + 1]) * anchor_scale[anchor_idx  * 3 + 1],
      sigmoid(cov_mlp_output[i * 7 + 2]) * anchor_scale[anchor_idx  * 3 + 2]
    ];
    const quat_vec = normalize([
      cov_mlp_output[i * 7 + 3],
      cov_mlp_output[i * 7 + 4],
      cov_mlp_output[i * 7 + 5],
      cov_mlp_output[i * 7 + 6]
    ]);
    const quat = makeQuaternionFromVec(quat_vec);
    const opacity = opacity_mlp_output[i];
    const color = [
      color_mlp_output[i * 3 + 0],
      color_mlp_output[i * 3 + 1],
      color_mlp_output[i * 3 + 2]
    ];
    const mtx = makeMatrix4(center, quat, scale, color, opacity);
    mtx_all.set(mtx, idx * 16);
  }
}
  
onmessage = async (e) => {
  if (e.data.type === 'init') {
    vertexCount = await init();
    postMessage({ type: 'initDone', vertexCount});
  } else if (e.data.type === 'compute') {
    const camera_position = e.data.camera_position;
    const camera_projectionMatrix = e.data.camera_projectionMatrix;
    const camera_viewMatrix = e.data.camera_viewMatrix;
    const window_size = e.data.window_size;
    const view = new Float32Array(e.data.view);
    await computeMtx(camera_position, camera_projectionMatrix, camera_viewMatrix, window_size);
    const newmatrices = Float32Array.from(mtx_all);
    postMessage({ type: 'computeDone', newmatrices }, [newmatrices.buffer]);
  }
};