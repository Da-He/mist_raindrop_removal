'''
The code was modified based on the repository https://github.com/zhilin007/FFA-Net [1].
[1] X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia, Ffa-net: Feature fusion attention network for single image dehazing, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020, pp. 11908–11915.
'''

import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from default_PA import MODEL_PA
from PerceptualLoss import LossNetwork as PerLoss
