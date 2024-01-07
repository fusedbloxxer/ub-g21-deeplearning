import argparse as ap

import gic


root_parser = ap.ArgumentParser(prog='gic', description=gic.PROJECT_NAME + ' CLI')
root_parser.parse_args()



# Allow training for each model
# Allow validation for each model
# Allow optimization for each model
# Allow seed, path, submission name
