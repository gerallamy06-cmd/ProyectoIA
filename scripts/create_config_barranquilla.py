import yaml
config_src = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\configs\eval_skysense_pp_flood3i.yml"
config_dst = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\configs\eval_skysense_pp_barranquilla.yml"
with open(config_src) as f:
    config = yaml.safe_load(f)
config['task_attributes']['segmentation']['dataset_attributes']['few_shot_flood_segmentation'].update({
    'data_root_dir': 'eval_datasets/barranquilla_flood',
    'data_txt': 'eval_datasets/barranquilla_flood/val.txt',
    'img_dir': 'eval_datasets/barranquilla_flood/Images',
    'tgt_dir': 'eval_datasets/barranquilla_flood/Semantic_mask',
    'image_size': {'hr': (512, 512), 's2': (16, 16), 's1': (16, 16), 'anno': (512, 512)}
})
config['model_attributes']['SkySensePP']['rec_head_hr']['num_classes'] = 2
with open(config_dst, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Config created:', config_dst)