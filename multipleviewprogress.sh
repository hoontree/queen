set -e

workdir=$1
data_dir=./data/dynerf/n3dv/$workdir

if [ ! -d "$data_dir" ]; then
    echo "Error: $data_dir does not exist"
    exit 1
fi

if [ -d ./colmap_tmp ]; then
    echo "Error: ./colmap_tmp already exists. Remove it before running."
    exit 1
fi

python scripts/extractimages.py dynerf/n3dv/$workdir

colmap feature_extractor --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images  --FeatureExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
colmap exhaustive_matcher --database_path ./colmap_tmp/database.db
mkdir ./colmap_tmp/sparse
colmap mapper --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images --output_path ./colmap_tmp/sparse

# Back up existing sparse_/ if present, then copy new sparse output
if [ -d "$data_dir/sparse_" ]; then
    mv "$data_dir/sparse_" "$data_dir/sparse_.bak"
fi
mkdir -p "$data_dir/sparse_"
cp -r ./colmap_tmp/sparse/0/* "$data_dir/sparse_"

mkdir ./colmap_tmp/dense
colmap image_undistorter --image_path ./colmap_tmp/images --input_path ./colmap_tmp/sparse/0 --output_path ./colmap_tmp/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --input_type geometric --output_path ./colmap_tmp/dense/fused.ply

# Back up existing point cloud if present, then write the new one with the loader-expected name
if [ -f "$data_dir/points3D_downsample2.ply" ]; then
    mv "$data_dir/points3D_downsample2.ply" "$data_dir/points3D_downsample2.ply.bak"
fi
python scripts/downsample_point.py ./colmap_tmp/dense/fused.ply "$data_dir/points3D_downsample2.ply"

# Generate poses_bounds.npy via LLFF, but DO NOT overwrite the dataset's existing one.
# The original poses_bounds.npy that ships with the dataset is the ground-truth pose file
# the loader reads (scene/dataset_readers.py:424). We only save the COLMAP-derived poses
# alongside for inspection.
if [ ! -d ./LLFF ]; then
    git clone https://github.com/Fyusion/LLFF.git
fi
pip install scikit-image
python LLFF/imgs2poses.py ./colmap_tmp/

cp ./colmap_tmp/poses_bounds.npy "$data_dir/poses_bounds_colmap.npy"

rm -rf ./colmap_tmp
rm -rf ./LLFF
