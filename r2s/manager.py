"""Manager of tropical cyclone ocean surface wind reanalysis system.

"""
import datetime
import getopt
import logging
import os
import sys

import checker
import regression
import cwind
import stdmet
import sfmr
import compare_offshore
import ibtracs
import era5
import hwind
import load_configs
import utils
import satel_scs
import grid
import coverage
import isd
import reg_scs
import ccmp
import compare_tc
import match_era5_smap
import validate
import statistic_ibtracs as sta_ibtracs
import statistic_era5_smap_sfmr as sta_era5_smap
import smart_compare
import merra2
import match_era5_sfmr
import combine_table
import classify
import simulate

unixOptions = 'p:r:eg:c:siv:k'
# gnuOptions = ['match_smap', 'reg-dnn', 'reg-xgb', 'reg-dt',
#               'reg-hist', 'reg-normalization', 'compare',
#               'sfmr', 'ibtracs-wp', 'ibtracs-na']
gnuOptions = ['period=', 'region=', 'basin=', 'match_smap', 'reg=',
              'compare=', 'sfmr', 'ibtracs', 'validate=', 'check',
              'sta_ibtracs', 'sta_era5_smap=', 'smart_compare',
              'merra2', 'match_sfmr', 'combine', 'tag=',
              'classify=', 'smogn_target=', 'draw_sfmr=',
              'max_windspd=', 'force_align_smap=',
              'interval=', 'simulate=']

def work_flow():
    """The work flow of blending several TC OSW.
    """
    load_configs.setup_logging()
    logger = logging.getLogger(__name__)
    # CONFIG
    try:
        CONFIG = load_configs.load_config()
    except Exception as msg:
        logger.exception('Exception occurred when loading config.')
    os.makedirs(CONFIG['logging']['dir'], exist_ok=True)

    # read commandline arguments, first
    full_cmd_arguments = sys.argv

    # - further arguments
    argument_list = full_cmd_arguments[1:]

    try:
        arguments, values = getopt.getopt(argument_list, '',
                                          gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    input_custom_period = False
    input_custom_region = False
    specify_basin = False
    basin = None
    do_match_smap = False
    do_regression = False
    reg_instructions = None
    smogn_target = None
    interval = None
    do_simulate = False
    do_classify = False
    classify_instruction = None
    tag = None
    do_compare = False
    draw_sfmr = False
    max_windspd = None
    force_align_smap = False
    do_sfmr = False
    sfmr_instructions = None
    do_ibtracs = False
    ibtracs_instructions = None
    do_validation = False
    do_check = False
    do_sta_ibtracs = False
    do_sta_era5_smap = False
    do_smart_compare = False
    do_merra2 = False
    do_match_sfmr = False
    do_combine = False
    # evaluate given options
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--period'):
            input_custom_period = True
            period_parts = current_value.split(',')
            if len(period_parts) != 2:
                logger.error((f"""Inputted period is wrong: """
                              f"""need 2 parameters"""))
        elif current_argument in ('-r', '--region'):
            input_custom_region = True
            region_parts = current_value.split(',')
            if len(region_parts) != 4:
                logger.error((f"""Inputted region is wrong: """
                              f"""need 4 parameters"""))
        elif current_argument in ('-b', '--basin'):
            specify_basin = True
            basin_parts = current_value.split(',')
            if len(basin_parts) != 1:
                logger.error((f"""Inputted basin is wrong: """
                              f"""must 1 parameters"""))
            basin = basin_parts[0]
        elif current_argument in ('-e', '--match_smap'):
            do_match_smap = True
        elif current_argument in ('-g', '--reg'):
            do_regression = True
            reg_instructions = current_value.split(',')
        elif current_argument in ('--smogn_target'):
            smogn_target = current_value.split(',')[0]
        elif current_argument in ('--interval'):
            interval = current_value.split(',')[:2]
        elif current_argument in ('--simulate'):
            do_simulate = True
            simulate_instructions = current_value.split(',')
        elif current_argument in ('--classify'):
            do_classify = True
            classify_instructions = current_value.split(',')
        elif current_argument in ('--tag'):
            tag = current_value.split(',')[0]
        elif current_argument in ('-c', '--compare'):
            do_compare = True
            compare_instructions = current_value.split(',')
        elif current_argument in ('--draw_sfmr'):
            head = current_value.split(',')[0]
            if head == 'True':
                draw_sfmr = True
            elif head == 'False':
                draw_sfmr = False
            else:
                logger.error('draw_sfmr must be "True" or "False"')
                sys.exit(1)
        elif current_argument in ('--max_windspd'):
            head = current_value.split(',')[0]
            max_windspd = float(head)
        elif current_argument in ('--force_align_smap'):
            head = current_value.split(',')[0]
            if head == 'True':
                force_align_smap = True
            elif head == 'False':
                force_align_smap = False
            else:
                logger.error('force_align_smap must be "True" or "False"')
                sys.exit(1)
        elif current_argument in ('-s', '--sfmr'):
            do_sfmr = True
        elif current_argument in ('-i', '--ibtracs'):
            do_ibtracs = True
        elif current_argument in ('-v', '--validate'):
            do_validation = True
            validate_instructions = current_value
        elif current_argument in ('-k', '--check'):
            do_check = True
        elif current_argument in ('--sta_ibtracs'):
            do_sta_ibtracs = True
        elif current_argument in ('--sta_era5_smap'):
            do_sta_era5_smap = True
            sources = current_value.split(',')
        elif current_argument in ('--smart_compare'):
            do_smart_compare = True
        elif current_argument in ('--merra2'):
            do_merra2 = True
        elif current_argument in ('--match_sfmr'):
            do_match_sfmr = True
        elif current_argument in ('--combine'):
            do_combine = True

    if not specify_basin:
        logger.error('Must specify basin')
        exit()

    if input_custom_period:
        # Period parts
        # yyyy-mm-dd-HH-MM-SS
        period = [
            datetime.datetime.strptime(period_parts[0],
                                       '%Y-%m-%d-%H-%M-%S'),
            datetime.datetime.strptime(period_parts[1],
                                       '%Y-%m-%d-%H-%M-%S')
        ]
    else:
        period = [datetime.datetime(2015, 4, 1, 0, 0, 0),
                  datetime.datetime.now()]
    train_test_split_dt = datetime.datetime(2019, 1, 1, 0, 0, 0)

    if input_custom_region:
        # Area parts
        custom_region = []
        for part in region_parts:
            custom_region.append(float(part))
    else:
        region = [-90, 90, 0, 360]

    # Period
    logger.info(f'Period: {period}')
    # Region
    logger.info(f'Region: {region}')
    # MySQL Server root password
    passwd = '399710'
    # Download and read
    try:
        if do_combine:
            combine_table.TableCombiner(CONFIG, period, region, basin,
                                        passwd)
        if do_match_sfmr:
            match_era5_sfmr.matchManager(CONFIG, period, region, basin,
                                         passwd, False)
        if do_merra2:
            merra2.MERRA2Manager(CONFIG, period, False)
        if do_smart_compare:
            smart_compare.SmartComparer(CONFIG, period, basin, passwd)
        if do_sta_era5_smap:
            sta_era5_smap.Statisticer(CONFIG, period, basin, sources,
                                      passwd)
        if do_sta_ibtracs:
            sta_ibtracs.Statisticer(CONFIG, period, basin, passwd)
        if do_check:
            checker.Checker(CONFIG)
        if do_validation:
            validate.ValidationManager(CONFIG, period, basin,
                                       validate_instructions)
        if do_match_smap:
            match_era5_smap.matchManager(
                CONFIG, period, region, basin, passwd, False, work=True)
        if do_classify:
            classify.Classifier(
                CONFIG, period, train_test_split_dt, region, basin,
                passwd, False, classify_instructions, smogn_target)
        if do_simulate:
            simulate.TCSimulator(
                CONFIG, period, region, basin, passwd, False,
                simulate_instructions)
        if do_regression:
            # if tag is None:
            #     logger.error('No model tag')
            #     exit()
            regression.Regression(
                CONFIG, period, train_test_split_dt, region, basin,
                passwd, False, reg_instructions, smogn_target, tag)
        if do_compare:
            # if ('smap_prediction' in compare_instructions
            #         and tag is None):
            #     logger.error('No model tag')
            #     exit()
            compare_tc.TCComparer(CONFIG, period, region, basin,
                                  passwd, False, compare_instructions,
                                  draw_sfmr, max_windspd,
                                  force_align_smap)
                                  # tag)
        # ccmp_ = ccmp.CCMPManager(CONFIG, period, region, passwd,
        #                          work_mode='fetch_and_compare')
        # era5_ = era5.ERA5Manager(CONFIG, period, region, passwd,
        #                          work=True, save_disk=False, 'scs',
        #                          'surface_all_vars')
        # isd_ = isd.ISDManager(CONFIG, period, region, passwd,
        #                       work_mode='fetch_and_read')
        # grid_ = grid.GridManager(CONFIG, region, passwd, run=True)
        # satel_scs_ = satel_scs.SCSSatelManager(CONFIG, period,
        #                                        region, passwd,
        #                                        save_disk=False,
        #                                        work=True)
        # coverage_ = coverage.CoverageManager(CONFIG, period,
        #                                      region, passwd)
        if do_ibtracs:
            ibtracs_ = ibtracs.IBTrACSManager(CONFIG, period, region,
                                              basin, passwd)
        # cwind_ = cwind.CwindManager(CONFIG, period, region, passwd)
        # stdmet_ = stdmet.StdmetManager(CONFIG, period, region, passwd)
        if do_sfmr:
            sfmr_ = sfmr.SfmrManager(CONFIG, period, region, passwd)
        # satel_ = satel.SatelManager(CONFIG, period, region, passwd,
        #                             save_disk=False)
        # compare_ = compare_offshore.CompareCCMPWithInStu(
        #     CONFIG, period, region, passwd)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    try:
        # new_reg = reg_scs.NewReg(CONFIG, period, test_period,
        #                          region, passwd, save_disk=True)
        # ibtracs_ = ibtracs.IBTrACSManager(CONFIG, test_period,
        #                                   region, passwd)
        # hwind_ = hwind.HWindManager(CONFIG, test_period, region, passwd)
        # era5_ = era5.ERA5Manager(CONFIG, test_period, region, passwd,
        #                          work=True, save_disk=False)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    logger.info('SWFusion complete.')
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
