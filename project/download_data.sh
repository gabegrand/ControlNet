mkdir data
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/training/fill50k.zip 
unzip fill50k.zip
mv fill50k data
rm fill50k.zip
python split_data.py

cp -r ../models ./ 
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
mv v1-5-pruned.ckpt models/v1-5-pruned.ckpt
python ../tool_add_control.py models/v1-5-pruned.ckpt models/control_sd15_ini.ckpt

