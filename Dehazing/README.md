### Training
 1. Download the training and testing dataset
 2. To train VIVNet on the ITS dataset, run
```
cd VIVNet
CONFIG=Desnowing/Options/ITS.yml python train_VIVNet.py
 ```


### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data ITS
``` 
to generate resulting images
3. Run 
```
python calculate_psnr_ssim.py --data ITS
``` 
to obtain the numerical results

``ITS`` can be replaced by ``Haze4k``, ``Dense-Haze``, ``NH-HAZE``,  
``SateHaze1k-Thin``, ``SateHaze1k-Moderate``, ``SateHaze1k-Thick`` for other datasets

