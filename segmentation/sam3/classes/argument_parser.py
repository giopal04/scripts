import sys
import argparse
from distutils.util import strtobool

def define_boolean_argument(parser, var_name, cmd_name, dst_variable, help_str, default, debug = False):
	parser.add_argument('--'+cmd_name, dest=dst_variable, action='store_true', help=help_str)
	parser.add_argument('--no-'+cmd_name, dest=dst_variable, action='store_false')
	#parser.set_defaults(show_label=True)
	cmd_fstring = f'parser.set_defaults({var_name}={bool(strtobool(str(default)))})'
	if debug:
		print(cmd_fstring)
	exec(cmd_fstring)
	if debug:
		print(f'{var_name = } - {bool(strtobool(str(default))) = }')
		#exec("print(%s)" % var_name)

def var2opt(dst_variable):
	if '-' in dst_variable:
		raise Exception(f'Error parsing variable: {dst_variable} - Boolean variable names cannot contain dashes `-`')
	return dst_variable, dst_variable.replace('_', '-'), dst_variable
