  


### Overview
Captures a person’s silhouette via webcam, processes it with YOLOv8 segmentation, and sends it to a remote machine (lamb) where a Stable Diffusion image-to-image model (SDXL) transforms the shape into an abstract visual, such as a liquid mercury wave. The resulting image is then sent back and displayed immediately.

The loop runs live:
    - Webcam captures a person’s pose every few seconds
    - A segmented mask is SCP’d to the remote SDXL machine
    - The remote system generates a matching abstract image
    - The output is SCP’d back and displayed via projector


#### Notes to Self:  
- cd gen_dance/
- source sdxl_env/bin/activate
- ssh bsf0891@lamb.mech.northwestern.edu while on eduroam wifi
- Local_live_loop.py 
- Remote_generate_loop.py
- scp remote_generate_loop.py	bsf0891@129.105.69.10:~

