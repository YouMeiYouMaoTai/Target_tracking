from PIL import Image
import numpy as np
def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
	im = Image.fromarray(arr, mode=mode)
	ts = type(size)
	if np.issubdtype(ts, np.signedinteger):
		percent = size / 100.0
		size = tuple((np.array(im.size)*percent).astype(int))
	elif np.issubdtype(type(size), np.floating):
		size = tuple((np.array(im.size)*size).astype(int))
	else:
		size = (size[1], size[0])
	func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
	imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
	return np.array(imnew)