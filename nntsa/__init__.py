from . import models, stats, tsa

import logging

logging.basicConfig(level=logging.INFO)  # set & initiate globally all handlers (to block unwanted logger)
logger = logging.getLogger('nntsa')  # a named logger
logger.getEffectiveLevel()  # -> becomes 20 after basicConfig()
logger.setLevel(logging.INFO)  # level of this logger, overwrite