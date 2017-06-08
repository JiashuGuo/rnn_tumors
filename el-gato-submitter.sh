###========================================
#!/bin/bash

#BSUB -n 1
#BSUB -R gpu
#BSUB -R "span[ptile=2]"
#BSUB -x
#BSUB -o Olsf.out
#BSUB -e Elsf.err
#BSUB -q "standard"
#BSUB -J 3D_Opt_F
#-----------------------------------------

python keras-rnn-biodata.py > output.txt


