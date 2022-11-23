# Explanation
To verify the effectiveness of our method, we provide the model trained on the Office-Home and Office31 datasets along with the associated verification code. And we will release all source code if the paper is accepted.
# Environment
- Python 3.6
- cuda9.2 + cudnn7.6.3
- GPU: two GeForce GTX 1080 Ti 
- pytorch 1.7.0

# Requirements
    conda install --yes --file requirements.txt

# Dataset

Download the dataset Office-Home:（https://pan.baidu.com/s/15NzPj74XMDG0fLbyvgkYjA?pwd=ehgi  ,Extract code:ehgi)
Download the dataset Office31:（https://pan.baidu.com/s/1RWiluvan05EDjrkbayzmMQ?pwd=fwan   ,Extract code:fwan )

Data Folder structure: 
```
Your dataset DIR:
|-OfficeHome
| |-Art
| |-Product
| |-...
```
You need to modify 'root' in the config file './config/oh_pcs.json'.

You need to move the pretrained model to './model_weights/'

# Validation
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

office31 model weights download: （https://pan.baidu.com/s/1OTjyjSOsgJHFihCPskW5bQ?pwd=05ed ,Extract code:05ed )

Validation on Office31:


```
A->W
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=amazon target=webcam init_weight='./model_weights/am_we.pth' 

A->D
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=amazon target=dslr init_weight='./model_weights/am_ds.pth' 

W->D
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=webcam target=dslr init_weight='./model_weights/we_ds.pth' 

W->A
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=webcam target=amazon init_weight='./model_weights/we_am.pth' 

D->A
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=dslr target=amazon init_weight='./model_weights/ds_am.pth' 

D->W
CUDA_VISIBLE_DEVICES=2,3 python office_validation.py note=validation multi_gpu=1 source=dslr target=webcam init_weight='./model_weights/ds_we.pth' 

```


The final results  will be saved in the './snapshot/validation/result.txt. '


