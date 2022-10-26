### Explanation
To verify the effectiveness of our method, we provide the model trained on the Office-Home dataset along with the associated verification code. And we will release all source code if the paper is accepted.
### Environment
- Python 3.6
- cuda9.2 + cudnn7.6.3
- GPU: GeForce GTX 1080 Ti
- pytorch 1.7.0

### Requirements
    conda install --yes --file requirements.txt

### Dataset

Download the dataset Office-Home
Data Folder structure: 
```
Your dataset DIR:
|-OfficeHome
| |-Art
| |-Product
| |-...
```
You need to modify 'root' in the config file './config/oh_pcs.json'
You need to move the pretrained model to './model_weights/'

## Validation

Validation on Office-Home: 
```
P->R
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Product target=RealWorld init_weight='./model_weights/Pr_Rw.pth' 

R->P
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=RealWorld target=Product init_weight='./model_weights/Rw_Pr.pth' 

C->R
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Clipart target=RealWorld init_weight='./model_weights/Cl_Rw.pth' 

C->P
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Clipart target=Product init_weight='./model_weights/Cl_Pr.pth' 

C->A
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Clipart target=Art init_weight='./model_weights/Cl_Ar.pth' 

```

The final results  will be saved in the './snapshot/validation/result.txt. '

