gpu_id=$1
scene_name=$2
export CUDA_VISIBLE_DEVICES=$gpu_id&&mamba run -n 3dgs python render.py --scene $scene_name -m output/$scene_name --config configs/n3dv.yaml
wait
export CUDA_VISIBLE_DEVICES=$gpu_id&&mamba run -n 3dgs python metrics.py -m output/$scene_name 
echo "Done"