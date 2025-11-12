conda activate op

python operating_platform/core/inference.py \
    --robot.type=so101 \
    --inference.single_task="start and test so101 arm." \
    --inference.dataset.repo_id="so101-test" \
    --policy.path="outputs/train/act_so101_test/checkpoints/010000/pretrained_model"