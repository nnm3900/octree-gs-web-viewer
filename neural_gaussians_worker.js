importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
//importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js');
//ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
//ort.env.wasm.numThreads = 1;

let anchor_pos, anchor_scale, anchor_features, gsplat_positions, anchor_feature_with_camera;
let anchorCount, vertexCount;
let color_mlp_session, opacity_mlp_session, cov_mlp_session;

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
  const buffer_gsplat_positions = await loadFile("./converted/gsplat_positions.bin");
  gsplat_positions = new Float32Array(buffer_gsplat_positions);

  anchorCount = 10000;//anchor_features.length / 32;
  vertexCount = anchorCount * 10;

  anchor_feature_with_camera = new Float32Array(35 * anchorCount);
  for (let i = 0; i < anchorCount; i++) {
    const anchor_feature = anchor_features.slice(i * 32, (i + 1) * 32);
    anchor_feature_with_camera.set(anchor_feature, i * 35);
  }

  const option = { executionProviders: ['webgl'] };
  color_mlp_session = await ort.InferenceSession.create('./converted/color_mlp.ort', option);
  opacity_mlp_session = await ort.InferenceSession.create('./converted/opacity_mlp.ort', option);
  cov_mlp_session = await ort.InferenceSession.create('./converted/cov_mlp.ort', option);

  postMessage({ type: 'initDone', vertexCount});
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
  
  async function computeMtx(camera_position) {

    for (let i = 0; i < anchorCount; i++) {
      const dx = anchor_pos[i * 3 + 0] - camera_position.x;
      const dy = anchor_pos[i * 3 + 1] - camera_position.y;
      const dz = anchor_pos[i * 3 + 2] + camera_position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      anchor_feature_with_camera[i * 35 + 32] = dx / distance;
      anchor_feature_with_camera[i * 35 + 33] = dy / distance;
      anchor_feature_with_camera[i * 35 + 34] = dz / distance;
    }
  
    const inputTensor = new ort.Tensor('float32', anchor_feature_with_camera, [anchorCount, 1, 1, 35]);
    const color_mlp_outputMap = await color_mlp_session.run({ 'input': inputTensor });
    const color_mlp_outputTensor = color_mlp_outputMap.output;
    const opacity_mlp_outputMap = await opacity_mlp_session.run({ 'input': inputTensor });
    const opacity_mlp_outputTensor = opacity_mlp_outputMap.output;
    const cov_mlp_outputMap = await cov_mlp_session.run({ 'input': inputTensor });
    const cov_mlp_outputTensor = cov_mlp_outputMap.output;
  
    const mtx_all = new Float32Array(vertexCount * 16);
    for (let i = 0; i < vertexCount; i++) {
      const center = [
        gsplat_positions[3 * i + 0],
        gsplat_positions[3 * i + 1],
        -gsplat_positions[3 * i + 2]
      ];
  
      const scale = [
        sigmoid(cov_mlp_outputTensor.data[i * 7 + 0]) * anchor_scale[Math.floor(i / 10) * 3 + 0],
        sigmoid(cov_mlp_outputTensor.data[i * 7 + 1]) * anchor_scale[Math.floor(i / 10) * 3 + 1],
        sigmoid(cov_mlp_outputTensor.data[i * 7 + 2]) * anchor_scale[Math.floor(i / 10) * 3 + 2]
      ];
  
      const quat_vec = normalize([
        cov_mlp_outputTensor.data[i * 7 + 3],
        cov_mlp_outputTensor.data[i * 7 + 4],
        cov_mlp_outputTensor.data[i * 7 + 5],
        cov_mlp_outputTensor.data[i * 7 + 6]
      ]);
      const quat = makeQuaternionFromVec(quat_vec);
  
      const opacity = opacity_mlp_outputTensor.data[i];
      const color = [
        color_mlp_outputTensor.data[i * 3 + 0],
        color_mlp_outputTensor.data[i * 3 + 1],
        color_mlp_outputTensor.data[i * 3 + 2]
      ];
  
      const mtx = makeMatrix4(center, quat, scale, color, opacity);

      mtx_all.set(mtx, i * 16);
    }
    return mtx_all;
  }
  
  onmessage = async (e) => {
    if (e.data.type === 'init') {
      await init();
    } else if (e.data.type === 'compute') {
      const { camera_position } = e.data;
      const newmatrices = await computeMtx(camera_position);
      postMessage({ type: 'computeDone', newmatrices }, [newmatrices.buffer]);
    }
  };