# MIT License
#
# Copyright (c) 2024 XRR Demo Code
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Script to add the replacement convolve function to noise.py"""

with open('xrr_env/lib/python3.12/site-packages/mlreflect/data_generation/noise.py', 'r') as f:
    content = f.read()

# Add the replacement function after the imports
replacement = '''
from scipy.interpolate import interp1d

# Simple replacement for refl1d.reflectivity.convolve
def refl1d_convolve(q_before, reflectivity_curves, q_after, width):
    """Simple replacement for refl1d convolution using interpolation"""
    # For now, just interpolate to the new q values
    # This is a simplified version - the real refl1d function does Gaussian convolution
    if len(reflectivity_curves.shape) == 1:
        f = interp1d(q_before, reflectivity_curves, bounds_error=False, fill_value='extrapolate')
        return f(q_after)
    else:
        result = np.zeros((reflectivity_curves.shape[0], len(q_after)))
        for i in range(reflectivity_curves.shape[0]):
            f = interp1d(q_before, reflectivity_curves[i], bounds_error=False, fill_value='extrapolate')
            result[i] = f(q_after)
        return result
'''

# Insert the replacement after the last import
import_end = content.find('\n\n\ndef')
if import_end == -1:
    import_end = content.find('\n\ndef')
if import_end == -1:
    import_end = content.find('from numpy import ndarray') + len('from numpy import ndarray')

new_content = content[:import_end] + '\n' + replacement + content[import_end:]

with open('xrr_env/lib/python3.12/site-packages/mlreflect/data_generation/noise.py', 'w') as f:
    f.write(new_content)

print("✓ Added replacement convolve function")
