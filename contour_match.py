def find_curvatures(contour, diff=5):
	curvatures = []
	for i in range(len(contour)):
		x = contour[(i - diff) % len(contour)][0]
		y = contour[i][0]
		z = contour[(i + diff) % len(contour)][0]
		
		x = x[0] + x[1] * 1j
		y = y[0] + y[1] * 1j
		z = z[0] + z[1] * 1j

		w = z-x
		w /= y-x
		c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
		r = abs(c + x)
		curvature = 1. / r
		curvatures.append(curvature)
	return curvatures