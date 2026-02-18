path="$DIR/logs/abTokens/version_0/checkpoints/last.ckpt"
data_dir=$DIR/antibody_data/rabd/multimer
experiment_name=abTokens

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python ./src/script/run_reconstruction.py --config-name=reconstruction.yaml \
    tokenizer=WrappedESM3Tokenizer \
    trainer.devices=[0] \
    optimization.micro_batch_size=1 \
    data.pdb_data_dir=${pdb_data_dir} \
    data.filter_length=100000 \
    model.class_name=VQVAEModelComplex \
    model.quantizer.use_linear_project=false \
    model.quantizer.freeze_codebook=false \
    model.encoder_attention=all \
    model.decoder_attention=all \
    model.ckpt_path=${path} \
    default_data_dir=${data_dir} \
    experiment_name=${experiment_name} \
    data.data_version=complex \
    model.save_path=$DIR/generated_pdbs/${experiment_name} \
    trainer.default_root_dir=$DIR/logs/