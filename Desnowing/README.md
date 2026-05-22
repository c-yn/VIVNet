### Training 
 1. Download the training and testing dataset
 2. To train VIVNet on the SRRS dataset, run
```
cd VIVNet
CONFIG=Desnowing/Options/SRRS.yml python train_VIVNet.py
 ```

on the Snow100K dataset, run
```
cd VIVNet
CONFIG=Desnowing/Options/Snow100K.yml python train_VIVNet.py
 ```


### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data CSD
python evaluation_from_data.py --data SRRS
python evaluation_from_data.py --data Snow100K
``` 
to generate resulting images
3. Run 
```
python calculate_psnr_ssim.py --data CSD
python calculate_psnr_ssim.py --data SRRS
python calculate_psnr_ssim.py --data Snow100K
``` 
to obtain the numerical results
