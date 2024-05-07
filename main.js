import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as ort from 'onnxruntime-web';

// Create the renderer
var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setClearColor(0xffffff, 0);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  200.0
);
camera.position.set(2.166491, 1.068512, 5.222922);
scene.add(camera);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.5;
const size = new THREE.Vector2();
renderer.getSize(size);
const focal = size.y / 2.0 / Math.tan(((camera.fov / 2.0) * Math.PI) / 180.0);
const geometry = new THREE.PlaneGeometry(4, 4);
const material = new THREE.ShaderMaterial({
  uniforms: {
    viewport: { value: new Float32Array([size.x, size.y]) },
    focal: { value: focal },
  },
  vertexShader: `varying vec4 vColor;
            varying vec2 vPosition;
            uniform vec2 viewport;
            uniform float focal;

            void main () {
                vec4 center = vec4(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2], 1);
                // Adjust View Pose
                mat4 adjViewMatrix = inverse(viewMatrix);
                adjViewMatrix[0][1] *= -1.0;
                adjViewMatrix[1][0] *= -1.0;
                adjViewMatrix[1][2] *= -1.0;
                adjViewMatrix[2][1] *= -1.0;
                adjViewMatrix[3][1] *= -1.0;
                adjViewMatrix = inverse(adjViewMatrix);
                mat4 modelView = adjViewMatrix * modelMatrix;

                vec4 camspace = modelView * center;
                vec4 pos2d = projectionMatrix * mat4(1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1)  * camspace;

                float bounds = 1.2 * pos2d.w;
                if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds
                    || pos2d.y < -bounds || pos2d.y > bounds) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                mat3 J = mat3(
                    focal / camspace.z, 0., -(focal * camspace.x) / (camspace.z * camspace.z), 
                    0., -focal / camspace.z, (focal * camspace.y) / (camspace.z * camspace.z), 
                    0., 0., 0.
                );

                mat3 W = transpose(mat3(modelView));
                mat3 T = W * J;
                mat3 cov = transpose(T) * mat3(instanceMatrix) * T;

                vec2 vCenter = vec2(pos2d) / pos2d.w;

                float diagonal1 = cov[0][0] + 0.3;
                float offDiagonal = cov[0][1];
                float diagonal2 = cov[1][1] + 0.3;

                float mid = 0.5 * (diagonal1 + diagonal2);
                float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
                float lambda1 = mid + radius;
                float lambda2 = max(mid - radius, 0.1);
                vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
                vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
                vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

                vColor = vec4(instanceMatrix[0][3], instanceMatrix[1][3], instanceMatrix[2][3], instanceMatrix[3][3]);
                vPosition = position.xy;

                gl_Position = vec4(
                    vCenter 
                        + position.x * v2 / viewport * 2.0 
                        + position.y * v1 / viewport * 2.0, 0.0, 1.0);
            }`,
  fragmentShader: `varying vec4 vColor;
            varying vec2 vPosition;

            void main () {
                float A = -dot(vPosition, vPosition);
                if (A < -4.0) discard;
                float B = exp(A) * vColor.a;
                gl_FragColor = vec4(B * vColor.rgb, B);
            }`,
});
material.blending = THREE.CustomBlending;
material.blendEquation = THREE.AddEquation;
material.blendSrc = THREE.OneMinusDstAlphaFactor;
material.blendDst = THREE.OneFactor;
material.blendSrcAlpha = THREE.OneMinusDstAlphaFactor;
material.blendDstAlpha = THREE.OneFactor;
material.depthTest = false;
material.needsUpdate = true;

window.addEventListener("resize", () => {
  renderer.getSize(size);
  const focal =
    size.y /
    2.0 /
    Math.tan(((camera.components.camera.data.fov / 2.0) * Math.PI) / 180.0);
  material.uniforms.viewport.value[0] = size.x;
  material.uniforms.viewport.value[1] = size.y;
  material.uniforms.focal.value = focal;
});

/*
async function main() {    
  const neural_gaussians_worker = new Worker('neural_gaussians_worker.js');
  const sort_gaussians_worker = new Worker('sort_gaussians_worker.js');
  let vertexCount;
  let matrices
  await new Promise((resolve) => {
    neural_gaussians_worker .onmessage = (e) => {
      if (e.data.type === 'initDone') {
        vertexCount = e.data.vertexCount;
        resolve();
      }
    };
    neural_gaussians_worker .postMessage({ type: 'init' });
  });
  await new Promise((resolve) => {
    neural_gaussians_worker .onmessage = (e) => {
      if (e.data.type === 'computeAndSortDone') {
        const sortedMatrices = new Float32Array(e.data.sortedMatrices);
        matrices = sortedMatrices;
        resolve();
      }
    };
    const view = new Float32Array([
      camera.matrixWorld.elements[2],
      camera.matrixWorld.elements[6],
      camera.matrixWorld.elements[10],
    ]);
    neural_gaussians_worker .postMessage({
      type: 'computeAndSort',
      camera_position: camera.position,
      view: view,
    });
  });
  const iMesh = new THREE.InstancedMesh(geometry, material, vertexCount);
  iMesh.frustumCulled = false;
  iMesh.instanceMatrix.array = matrices;
  iMesh.instanceMatrix.needsUpdate = true;
  scene.add(iMesh);

  let view = new Float32Array([
    camera.matrixWorld.elements[2],
    camera.matrixWorld.elements[6],
    camera.matrixWorld.elements[10],
  ]);
  let isWorkerBusy = false;
  worker.onmessage = (e) => {
    if (e.data.type === 'computeAndSortDone') {
      const sortedMatrices = new Float32Array(e.data.sortedMatrices);
      iMesh.instanceMatrix.array = sortedMatrices;
      iMesh.instanceMatrix.needsUpdate = true;
      isWorkerBusy = false;
    }
  };

  let prevCameraPosition = new THREE.Vector3();
  animate();
  function animate() {
    requestAnimationFrame(animate);
    if (!prevCameraPosition.equals(camera.position)) {
      if (!isWorkerBusy) {
        isWorkerBusy = true;
        const view = new Float32Array([
          camera.matrixWorld.elements[2],
          camera.matrixWorld.elements[6],
          camera.matrixWorld.elements[10],
        ]);
        worker.postMessage({
          type: 'computeAndSort',
          camera_position: camera.position,
          view: view,
        });
        prevCameraPosition.copy(camera.position);
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }
}
*/
async function main() {
  const neural_gaussians_worker = new Worker('neural_gaussians_worker.js');
  const sort_gaussians_worker = new Worker('sort_gaussians_worker.js');

  let vertexCount;
  let matrices;
  let sortReady = true;
  let neuralReady = true;
  await new Promise((resolve) => {
    neural_gaussians_worker.onmessage = (e) => {
      if (e.data.type === 'initDone') {
        vertexCount = e.data.vertexCount;
        resolve();
      }
    };
    neural_gaussians_worker.postMessage({ type: 'init' });
  });
  await new Promise((resolve) => {
    neural_gaussians_worker.onmessage = (e) => {
      if (e.data.type === 'computeDone') {
        matrices = new Float32Array(e.data.newmatrices);
        resolve();
      }
    };
    const camera_position = camera.position;
    neural_gaussians_worker.postMessage({
      type: 'compute',
      camera_position: camera_position,
    });
  });

  const iMesh = new THREE.InstancedMesh(geometry, material, vertexCount);
  iMesh.frustumCulled = false;
  iMesh.instanceMatrix.array = matrices;
  iMesh.instanceMatrix.needsUpdate = true;
  scene.add(iMesh);

  neural_gaussians_worker.onmessage = (e) => {
    if (e.data.type === 'computeDone') {
      matrices = new Float32Array(e.data.newmatrices);
      neuralReady = true;
    }
  };

  sort_gaussians_worker.onmessage = (e) => {
    const sortedMatrices = new Float32Array(e.data.sortedMatrices);
    iMesh.instanceMatrix.array = sortedMatrices;
    iMesh.instanceMatrix.needsUpdate = true;
    sortReady = true;
  };

  let prevCameraPosition = new THREE.Vector3();

  animate();
  function animate() {
    requestAnimationFrame(animate);
    if (!prevCameraPosition.equals(camera.position)) {
      if (neuralReady) {
        neuralReady = false;
        const camera_position = camera.position;
        neural_gaussians_worker.postMessage({
          type: 'compute',
          camera_position: camera_position,
        });
        prevCameraPosition.copy(camera.position);
      }
    }
    if (sortReady) {
      sortReady = false;
      const view = new Float32Array([camera.matrixWorld.elements[2], camera.matrixWorld.elements[6], camera.matrixWorld.elements[10],]);
      const matricesCopy = Float32Array.from(matrices);
      sort_gaussians_worker.postMessage({
        matrices: matricesCopy.buffer,
        view: view.buffer,},
        [matricesCopy.buffer, view.buffer]
      );
    }
    controls.update();
    renderer.render(scene, camera);
  }
}


main().catch((error) => {
  console.error(error);
});