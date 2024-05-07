# octree-gs-web-viewer
This page provides an exprimental web viewer for [Octree-GS [Ren 2024]](https://github.com/city-super/Octree-GS), which is a state-of-the-art extension of [Gaussian-Splatting [Kerbl 2023]](https://github.com/graphdeco-inria/gaussian-splatting). You van try online demo form [here](https://main--bejewelled-nougat-efb14c.netlify.app/). The code is based on [Gaussian-Splatting-WebViewers (gaussian_splatting_1)](https://github.com/akbartus/Gaussian-Splatting-WebViewers/tree/main), which is originally based on [ WebGL implementation of Gaussian Splatting](https://github.com/antimatter15/splat). 

## usage
`npm install`  
`npx webpack`  
`npx light-server -s . -p 8080`  
and open your browser with http://localhost:8080/  

## for your own data
1. Place the trained data of Octree-GS as follows  
root/  
├── gs-data/  
│   ├── point_cloud.ply  
│   ├── color_mlp.pt  
│   ├── cov_mlp.pt  
│   ├── opacity_mlp.pt  

2. Set up python environment
`conda env create --file environment.yml`  
`conda activate octree-`

3. Click on “run.bat” to convert the data

