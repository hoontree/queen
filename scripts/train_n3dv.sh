gpu_id=$1
scene_name=$2
init_from_ply=${3:-0}   # pass 1 to init frame 1 from <source_path>/init_3dgs.ply (skips static 3DGS init)

if [ "$init_from_ply" = "1" ]; then
    out_dir=output/plyinit/$scene_name
    ply_flag="--init_from_ply"
else
    out_dir=output/scratch/$scene_name
    ply_flag=""
fi

echo "init_from_ply=$init_from_ply -> output dir: $out_dir"

export CUDA_VISIBLE_DEVICES=$gpu_id&&mamba run -n 3dgs python train.py --scene $scene_name --config configs/n3dv.yaml --timed  -r 2 -m $out_dir --sh_degree 0 --save_iterations -1 $ply_flag
wait
export CUDA_VISIBLE_DEVICES=$gpu_id&&mamba run -n 3dgs python render.py --scene $scene_name -m $out_dir --config configs/n3dv.yaml --skip_train
wait
export CUDA_VISIBLE_DEVICES=$gpu_id&&mamba run -n 3dgs python metrics.py -m $out_dir
echo "Done"
