#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=TLlib
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=1-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/envs/TLlib/bin/activate

echo "change TORCH_HOME environment variable"
cd $SLURM_TMPDIR
cp -r ~/scratch/model_zoo .
export TORCH_HOME=$SLURM_TMPDIR/model_zoo

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/Transfer-Learning-Library .

echo "Copying the datasets"
date +"%T"
cd $SLURM_TMPDIR/Transfer-Learning-Library
cp -r ~/scratch/TLlib_Dataset .

echo "extract datasets"
date +"%T"
cd TLlib_Dataset
cd Office31
tar -xzf amazon.tgz
tar -xzf dslr.tgz
tar -xzf webcam.tgz
unzip image_list.zip

date +"%T"
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd Transfer-Learning-Library
cd examples/domain_adaptation/image_classification

CUDA_VISIBLE_DEVICES=0 python adda.py TLlib_Dataset/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/adda/Office31_A2W

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
# cd $SLURM_TMPDIR
# cp -r $SLURM_TMPDIR/Transfer-Learning-Library/logs/adda/Office31_A2W ~/scratch/Transfer-Learning-Library/logs/