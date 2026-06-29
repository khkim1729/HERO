# TN Staging and nnU-Net Segmentation after Excluding 48 Cases

This repository contains code, configuration files, metadata, and result summaries for TN staging and nnU-Net segmentation experiments after excluding 48 predefined cases.

cat > configs/tn_staging_predmask.yaml << 'EOF'
experiment_name: tn_staging_predmask_exclude48
task: tn_staging

data:
  exclusion_file: metadata/exclude_cases_48.txt
  split_file: metadata/split_final_exclude48.json
  mask_type: nnunet_predicted
  use_mask_as_input: true
  recommended_prediction_type: out_of_fold

target:
  t_stage_column: T_stage
  n_stage_column: N_stage

training:
  seed: 42
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001

output:
  result_file: results/tn_staging_predmask_metrics.json
  checkpoint_dir: checkpoints/tn_staging_predmask
