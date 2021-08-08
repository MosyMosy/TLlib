#!/usr/bin/env bash
# PACS
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/PACS -d PACS -s A C S -t P --freeze-bn --seed 0 --log logs/groupdro/PACS_P
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/PACS -d PACS -s P C S -t A --freeze-bn --seed 0 --log logs/groupdro/PACS_A
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/PACS -d PACS -s P A S -t C --freeze-bn --seed 0 --log logs/groupdro/PACS_C
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/PACS -d PACS -s P A C -t S --freeze-bn --seed 0 --log logs/groupdro/PACS_S
# Office-Home
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr --seed 0 --log logs/groupdro/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw --seed 0 --log logs/groupdro/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl --seed 0 --log logs/groupdro/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar --seed 0 --log logs/groupdro/OfficeHome_Ar
# DomainNet
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s i p q r s -t c -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s c p q r s -t i -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s c i q r s -t p -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s c i p r s -t q -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s c i p q s -t r -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/domainnet -d DomainNet -s c i p q r -t s -i 4000 -b 40 --lr 0.005 --seed 0 --log logs/groupdro/DomainNet_s
