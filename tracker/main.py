
#comment these two lines if it doesnt work
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import getopt
from point_to_define import display


name="sunny"


#FOR REGISTER (uses tracking for registration too)
# register = display.register(None, name)
# print(register)

#FOR AUTHENTICATE (logging in)
authenticate = display.authenticate(None, name)
print(authenticate)