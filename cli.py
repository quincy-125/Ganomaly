import argparse
import logging
import main
import sys

##############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--tf_record_dir",
                    dest='tf_record_dir',
                    required=True,
                    help="File path where TFRecords are located")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

if __name__ == '__main__':
    main.run(args)
