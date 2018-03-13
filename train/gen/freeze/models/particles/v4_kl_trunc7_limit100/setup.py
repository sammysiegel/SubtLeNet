
from subtlenet import config
from subtlenet.generators import gen as generator
from subtlenet.utils import set_processor
config.limit = 100
generator.truncate = 7
set_processor("gpu")
config.smear_params = None
