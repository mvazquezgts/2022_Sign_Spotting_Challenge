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

    path_experiment = '{}/SignSpotting/experiments/{}'.format(
        args.path_cslr, config['LABEL_EXPERIMENT'])
    path_experiment_queries = '{}/SignSpotting/experiments/{}'.format(
        args.path_cslr, config['LABEL_EXPERIMENT_QUERIES'])
    cmd_cd_islr = '{}/preprocessing'.format(args.path_islr)
    cmd_cd_sign_spotting = '{}/SignSpotting/src'.format(args.path_cslr)

    types_features = ''
    for type_features_idx in config['TYPE_FEATURES']:
        types_features = types_features + \
            '{}_{} '.format(type_features_idx, config['TYPE_KPS'])


    os.chdir(cmd_cd_sign_spotting)
    print ('cd {}'.format(cmd_cd_sign_spotting))
    
    if (config['PHASES_OP1'][2]):
        cmd = 'python detect_queries_on_videos.py --input_videos {}/B2_ISLR_output_RAW --input_queries {}/B2_ISLR_output_RAW --output {}/B2_SignSpotting_output --type_features {} --fps {} --file_query_limits {}'.format(
            path_experiment, path_experiment_queries, path_experiment, types_features, config['FPS'], config['QUERY_LIMITS'])
        execute_command(cmd, execute_commands)


if __name__ == '__main__':

    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--experiment', required=True, type=str)
    parser_main.add_argument('--path_cslr', required=True, type=str)
    parser_main.add_argument('--path_islr', required=True, type=str)
    parser_main.add_argument('--run', action='store_true')
    arg = parser_main.parse_args()
    main(arg)
    
    # python main_step2_track2.py --experiment OSLWL/EXPERIMENTO_OSLWL_VAL_SET --path_cslr '/tmp/ECCV2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr '/tmp/ECCV2022_Sign_Spotting_Challenge/ISLR_workspace' --run
    
