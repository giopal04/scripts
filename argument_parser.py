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

def argument_parser():

	parser = argparse.ArgumentParser(description='OAK-D video and depth h265 capture script')

	'''
color_resolutions = {
		'1080p': (1920,	1080, 60, dai.ColorCameraProperties.SensorResolution.THE_1080_P),
		'4K'   : (3840,	2160, 60, dai.ColorCameraProperties.SensorResolution.THE_4_K),
}
depth_resolutions = {
		'720p': (1280,	720, 60,  dai.MonoCameraProperties.SensorResolution.THE_720_P),
		'800p': (1280,	800, 60,  dai.MonoCameraProperties.SensorResolution.THE_800_P),
		'400p': (640,	400, 120, dai.MonoCameraProperties.SensorResolution.THE_400_P),
}
	'''

	# ---------------------
	# -- CAPTURE OPTIONS --
	# ---------------------
	define_boolean_argument(parser, *var2opt('disparity'),			'capture disparity instead of left/right streams'				, True)
	define_boolean_argument(parser, *var2opt('extended_disparity'),		'use extended disparity for closer distances'					, True)
	define_boolean_argument(parser, *var2opt('subpixel_disparity'),		'use extended disparity for closer distances'					, False)
	#define_boolean_argument(parser, *var2opt('leftright'),			'capture left/right instead of disparity stream'				, False)
	parser.add_argument('--confidence',  type=int, default=250.0,		help="set the confidence treshold for disparity")
	parser.add_argument('--color-resolution', default='1080p',		help='captured videos RGB resolution (1080p @ 60 FPS or 4K @ 60 FPS)')
	parser.add_argument('--depth-resolution', default='400p',		help='captured videos depth resolution (800p @ 60 FPS or 720p @ 60 FPS or 400p @ 120 FPS)')
	define_boolean_argument(parser, *var2opt('wls_filter'),			'apply WLS filter to disparity and save a separate CV2 video'			, False)
	define_boolean_argument(parser, *var2opt('rectified_right'),		'WLS filter is too slow on RPI4 (2 to 4 FPS), just record h265 rectified right for later use', True)
	define_boolean_argument(parser, *var2opt('rectified_left'),		'Also record h265 rectified left for later use with different disparity algorithms', False)
	define_boolean_argument(parser, *var2opt('rgb'),			'record RGB color h265 stream', True)
	parser.add_argument('--wls-max-queue',  type=int, default=10.0,		help="drop frames after wls queue has reached max_queue")
	parser.add_argument('--preview-max-queue',  type=int, default=10.0,	help="drop frames after preview queue has reached max_queue")

	# ------------------
	# -- OUTPUT FILES --
	# ------------------
	parser.add_argument('--output-dir', default='/mnt/btrfs-data',		help='captured videos output directory (default: /mnt/btrfs-data)')

	# --------------
	# -- HARDWARE --
	# --------------
	define_boolean_argument(parser, *var2opt('force_usb2'), 'force the OAK-D camera in USB2 mode (useful in low bitrate/low power scenarios)'		, False)

	# ---------------
	# -- DEBUGGING --
	# ---------------
	define_boolean_argument(parser, *var2opt('debug_img_sizes'),		'add debugging information about captured image sizes'				, False)
	define_boolean_argument(parser, *var2opt('debug_pipeline_types'),	'add debugging information about captured image types'				, False)
	define_boolean_argument(parser, *var2opt('debug_pipeline_steps'),	'add debugging information about capturing steps'				, False)
	define_boolean_argument(parser, *var2opt('debug_wls_threading'),	'add debugging information about WLS multithreaded filtering'			, False)
	define_boolean_argument(parser, *var2opt('debug_preview_threading'),	'add debugging information about threaded preview'				, False)

	# ------------------
	# -- VIEW OPTIONS --
	# ------------------
	define_boolean_argument(parser, *var2opt('show_preview'),		'global switch to show OpenCV windows with the captured images'			, False)
	define_boolean_argument(parser, *var2opt('show_wls_preview'),		'show host-side WLS-filtered disparity made with OpenCV (heavy)'		, False)
	define_boolean_argument(parser, *var2opt('write_preview'),		'write the captured images in OpenCV (JPG/PNG) format'				, False)
	define_boolean_argument(parser, *var2opt('write_wls_preview'),		'write host-side WLS filtering images made with OpenCV (JPG/PNG) format'	, False)
	define_boolean_argument(parser, *var2opt('show_rgb'),			'show preview of RGB image'							, False)
	define_boolean_argument(parser, *var2opt('show_colored_disp'),		'show colored preview of raw disparity'						, True)
	define_boolean_argument(parser, *var2opt('show_gray_disp'),		'show grayscale preview of raw disparity'					, False)
	define_boolean_argument(parser, *var2opt('show_rr_img'),		'show preview of (flipped) rectified right'					, True)
	define_boolean_argument(parser, *var2opt('show_th_disp'),		'show preview of tresholded disparity'						, False)
	parser.add_argument('--preview-downscale-factor',  type=int, default=1,	help="downscale all the previewed images by a factor of x (default: 1)")

	'''
	parser.set_defaults(show_fps=False)
	parser.set_defaults(show_frame_number=False)
	parser.set_defaults(debug_segments=False)
	parser.set_defaults(enable_ros=False)
	parser.set_defaults(batch_mode=False)
	parser.set_defaults(double_view=False)
	parser.set_defaults(demo_mode=False)
	parser.set_defaults(replay_mode=False)
	parser.set_defaults(do_inference=False)
	parser.set_defaults(do_inference_v2_seq=False)

	parser.set_defaults(show_only_scatter_points=False)
	parser.set_defaults(show_nodes=True)
	parser.set_defaults(show_axis=False)
	parser.set_defaults(show_color=True)
	parser.set_defaults(show_tips_traces=True)
	parser.set_defaults(show_edge_traces=False)
	'''

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)


	args = parser.parse_args()
	print('')
	print('')
	print(f'Python   received this arguments: {sys.argv}')
	print('')
	print(f'Argparse received this arguments: {args}')

	return args
