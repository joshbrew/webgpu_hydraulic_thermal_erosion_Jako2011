# WebGPU real time Hydraulic + Thermal erosion
<img width="800" alt="Screenshot 2026-04-09 145612" src="https://github.com/user-attachments/assets/2f41f99e-cb94-48fc-8844-99460129a27f" />

## Try it: [https://webgpuerosion.netlify.app/](https://webgpuerosion.netlify.app/)

Browser WebGPU implementation of "[Fast Hydraulic and Thermal Erosion on the GPU](https://old.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf)" by Balazs Jako (2011) with some modern tweaks for WebGPU and more control knobs.

Import images (heightmaps, DEM, anything) and run erosion simulations. Works on up to ~2048x2048 images in browser in the current version (we have a laptop RTX 4070). We tried higher res texture arrays but it looked bad using a chunking techniques to overcome buffer size limits (2GB in browser) on our last attempt. We might revisit it. I'm sure there's more performance optimization to do, and the SWE render is jank.

You can also paint terrain and spring sources onto your base map image. Click the canvas for camera controls, press esc to unlock your mouse after. 

### Build and run

Open this folder in the terminal.

`npm i -g tinybuild` then `tinybuild`

Chromium-based browsers recommended for the latest webgpu support.

## Results

The results are pretty good on default settings, which you can completely override in the demo UI:

Video 1: https://www.youtube.com/watch?v=qz4CPlcCj7E

Video 2: https://www.youtube.com/watch?v=o19LsY70Aco

Test image made in our [webgpu noise generator](https://github.com/joshbrew/webgpu_noise_compute_textures):

<img width="300" height="300" alt="noise_test" src="https://github.com/user-attachments/assets/94550ae7-bb3a-4f3c-9366-2e524d928909" />
<img width="300" height="300" alt="Screenshot 2026-04-08 172111" src="https://github.com/user-attachments/assets/b39114c2-5b55-462a-9c5c-51b71cb38adb" />
<img width="300" height="300" alt="Screenshot 2026-04-08 172106" src="https://github.com/user-attachments/assets/c19658c0-d506-437f-a500-8d833090eb1b" />

Another result on the same map:

<img width="300" height="300" alt="Screenshot 2026-04-09 145612" src="https://github.com/user-attachments/assets/aa8d260b-7793-4988-8378-3fa0d5cb169b" />
<img width="300" height="300" alt="Screenshot 2026-04-09 145435" src="https://github.com/user-attachments/assets/640892ac-2ebe-4f51-8b3a-5a5f41b1f72b" />
<img width="300" height="300" alt="Screenshot 2026-04-09 145456" src="https://github.com/user-attachments/assets/c854e42f-90d9-412d-96cb-7813437f6a0c" />

Another erosion result (same map, exported in-app):

<img width="300" height="300" alt="noise-main_export" src="https://github.com/user-attachments/assets/929fe38e-c4ea-4bef-9e1d-0616a8591706" />

More results:

<img width="300" height="300" alt="Screenshot 2026-04-08 165014" src="https://github.com/user-attachments/assets/8e22699c-df47-4ae1-b438-5380c780931e" />
<img width="300" height="300" alt="Screenshot 2026-04-08 163138" src="https://github.com/user-attachments/assets/82f4aa14-f01e-4645-b7ec-0cd70c8a95b2" />

And for fun using an image from our [webgpu fractal explorer](https://github.com/joshbrew/webgpu_fractals):

<img width="300" height="300" alt="fractal-1536-yfbvgz" src="https://github.com/user-attachments/assets/d26aac72-706d-4988-9bb7-2e2c5487edd8" />
<img width="300" height="300" alt="fractal-1536-yfbvgz_dem_export" src="https://github.com/user-attachments/assets/4de40ea7-a221-4a2a-9cf4-929cca641cc8" />
<img width="300" height="300" alt="fractal-1536-yfbvgz_dem_export (1)" src="https://github.com/user-attachments/assets/aed1d993-f820-4a92-8592-36fdfdecc278" />


### License

MIT

Go crazy! Feel free to make pull requests for feature improvements/additions.



