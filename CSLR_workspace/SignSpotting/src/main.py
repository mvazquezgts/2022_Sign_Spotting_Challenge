import argparse
import sys
import os
import yaml

def execute_command(cmd, execute_commands):
    print(cmd)
    if (execute_commands):
        os.system(cmd)

def main(args):
    print('main')

    execute_commands = args.run
    print('execute_commands: {}'.format(execute_commands))

    file_config = os.path.join('config', args.experiment, 'config.yaml')
    with open(file_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        print(config)

    path_sign_spotting = '{}/SignSpotting'.format(args.path_cslr)
    path_experiment = '{}/SignSpotting/experiments/{}'.format(
        args.path_cslr, config['LABEL_EXPERIMENT'])
    cmd_cd_islr = '{}/preprocessing'.format(args.path_islr)
    cmd_cd_sign_spotting = '{}/SignSpotting/src'.format(args.path_cslr)

    types_features = ''
    for type_features_idx in config['TYPE_FEATURES']:
        types_features = types_features + \
            '{}_{} '.format(type_features_idx, config['TYPE_KPS'])

    if (config['PHASES_PREPARATION'][0]):
        os.chdir(cmd_cd_islr)
        print ('cd {}'.format(cmd_cd_islr))
        
        cmd = 'python preprocess_video_playlist.py --input {}/{} --output {}/A1_preprocessing_videos --method CENTERED_CUT --resolution 512x512'.format(
            path_sign_spotting, config['VIDEO_FOLDER'], path_experiment)
        execute_command(cmd, execute_commands)

        os.chdir(cmd_cd_sign_spotting)
        print ('cd {}'.format(cmd_cd_sign_spotting))
        cmd = 'python preparate_elan_files.py --input {}/{} --output {}/A1_preprocessing_videos --labels {}'.format(
            path_sign_spotting, config['ELAN_FOLDER'], path_experiment, config['LABELS'])
        execute_command(cmd, execute_commands)

    if (config['PHASES_PREPARATION'][1]):
        os.chdir(cmd_cd_islr)
        print ('cd {}'.format(cmd_cd_islr))
        cmd = 'python generate_kps.py --input {}/A1_preprocessing_videos --output {}/A2_features/npy/kps'.format(
            path_experiment, path_experiment)
        execute_command(cmd, execute_commands)

        cmd = 'python generate_features.py --folder {}/A2_features/npy --type_kps {} --noframeslimit True'.format(
            path_experiment, config['TYPE_KPS'])
        execute_command(cmd, execute_commands)

    os.chdir(cmd_cd_sign_spotting)
    print ('cd {}'.format(cmd_cd_sign_spotting))
    
    if config['USE_CONTEXT']:
        use_context = '--use_context'
    else:
        use_context = ''

    if (config['PHASES_OP1'][0]):
        cmd = 'python generate_Windows.py --input {}/A2_features/npy --output {}/B1_generate_windows --type_features {} --window_size {} --window_stride {} '.format(
            path_experiment, path_experiment, types_features, config['WINDOWS_SIZE'], config['WINDOWS_STRIDE'])
        execute_command(cmd, execute_commands)

    if (config['PHASES_OP1'][1]):
        cmd = 'python generate_ISLR_output.py --input {}/B1_generate_windows --output {}/B2_ISLR_output --type_features {} --labels {} --folder_models {} --batch_size {} --device {} --raw'.format(
            path_experiment, path_experiment, types_features, config['LABELS'], config['FOLDER_MODELS'], config['BATCH_SIZE'], config['DEVICE_GPU'])
        execute_command(cmd, execute_commands)

    if (config['PHASES_OP1'][2]):
        cmd = 'python generate_SignSpotting_output.py --input {}/B2_ISLR_output --output {}/B3_SignSpotting_output --type_features {} --threshold {} --min_margin_is_sign {} --min_margin_is_same {} --fps {} --windows_size {} --window_stride {}'.format(
            path_experiment, path_experiment, types_features, config['THRESHOLD_POSITIVE'], config['MIN_MARGIN_IS_SIGN'], config['MIN_MARGIN_IS_SAME'], config['FPS'], config['WINDOWS_SIZE'], config['WINDOWS_STRIDE'])
        execute_command(cmd, execute_commands)

    if (config['PHASES_OP1'][3]):
        cmd = 'python filter_SignSpotting_output_ISLR.py --input_results {}/B3_SignSpotting_output --input_features {}/A2_features/npy --type_features {} --labels {} --threshold {} --folder_models {} --result_size_fixed {} {}'.format(
            path_experiment, path_experiment, types_features, config['LABELS'], config['THRESHOLD_FILTER'], config['FOLDER_FILTER_MODELS'], config['RESULT_SIZE_FIXED'], use_context)
        execute_command(cmd, execute_commands)
    
    select_output = ''
    if config['VIS']:
        select_output = select_output+'--vis '
    if config['METRIC']:
        select_output = select_output+'--metric '
    if config['ELAN']:
        select_output = select_output+'--subs '

    if (config['EVALUATION']):
        cmd = 'python generate_Evaluation.py --input {}/B3_SignSpotting_output --output {}/E1_evaluation_output --gt {}/A1_preprocessing_videos --labels {}/{} --threshold_iou_min {} --threshold_iou_max {} --threshold_iou_step {} {}'.format(
            path_experiment, path_experiment, path_experiment, path_sign_spotting, config['LABELS'], config['THRESHOLD_IOU_MIN'], config['THRESHOLD_IOU_MAX'], config['THRESHOLD_IOU_STEP'], select_output)
        execute_command(cmd, execute_commands)


if __name__ == '__main__':

    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--experiment', required=True, type=str)
    parser_main.add_argument('--path_cslr', required=True, type=str)
    parser_main.add_argument('--path_islr', required=True, type=str)
    parser_main.add_argument('--run', action='store_true')
    arg = parser_main.parse_args()
    main(arg)
    
    # python main.py --experiment MSSL/EXPERIMENTO_MSSL_TRAIN_SET --path_cslr '/home/temporal2/mvazquez/Challenge/baseline/CSLR_workspace' --path_islr '/home/temporal2/mvazquez/Challenge/baseline/ISLR_workspace'
    # python main.py --experiment MSSL/EXPERIMENTO_MSSL_VAL_SET --path_cslr '/home/temporal2/mvazquez/Challenge/baseline/CSLR_workspace' --path_islr '/home/temporal2/mvazquez/Challenge/baseline/ISLR_workspace'
    # python main.py --experiment OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET --path_cslr '/home/temporal2/mvazquez/Challenge/baseline/CSLR_workspace' --path_islr '/home/temporal2/mvazquez/Challenge/baseline/ISLR_workspace'
    # python main.py --experiment OSLWL/OSLWL_VAL_SET --path_cslr '/home/temporal2/mvazquez/Challenge/baseline/CSLR_workspace' --path_islr '/home/temporal2/mvazquez/Challenge/baseline/ISLR_workspace'
    