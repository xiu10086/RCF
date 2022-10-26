### Explanation
To verify the effectiveness of our method, we provide the model trained on the Office-Home dataset along with the associated verification code. And we will release all source code if the paper is accepted.
### Environment
- Python 3.6
- cuda9.2 + cudnn7.6.3
- GPU: two GeForce GTX 1080 Ti 
- pytorch 1.7.0

### Requirements
    conda install --yes --file requirements.txt

### Dataset

Download the dataset Office-Home (https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)

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
office-home model weights download: （https://pan.baidu.com/s/1vbMSYXMlJAbWJMuwrxNsrA?pwd=u4d3 ,Extract code:u4d3)
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

P->C
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Product target=Clipart init_weight='./model_weights/Pr_Cl.pth' 

P->A
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Product target=Art init_weight='./model_weights/Pr_Ar.pth' 

R->A
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=RealWorld target=Art init_weight='./model_weights/Rw_Ar.pth'

R->C
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=RealWorld target=Clipart init_weight='./model_weights/Rw_Cl.pth'

A->C
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Art target=Clipart init_weight='./model_weights/Ar_Cl.pth' 

A->P
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Art target=Product init_weight='./model_weights/Ar_Pr.pth'

A->R
CUDA_VISIBLE_DEVICES=2,3 python officehome_validation.py note=validation multi_gpu=1 source=Art target=RealWorld init_weight='./model_weights/Ar_Rw.pth' 

```

The final results  will be saved in the './snapshot/validation/result.txt. '

