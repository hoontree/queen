gpu_id=$1
scene_name=$2

# Run scratch (init_from_ply=0) first, then ply-init (init_from_ply=1).
bash scripts/train_n3dv.sh $gpu_id $scene_name 0
wait
bash scripts/train_n3dv.sh $gpu_id $scene_name 1
wait
echo "Both runs done"
